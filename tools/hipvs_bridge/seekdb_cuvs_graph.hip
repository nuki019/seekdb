// CAGRA graph construction on the GPU without NN-descent.
//
// The stock CAGRA build creates an approximate k-NN graph with NN-descent (about 100
// refinement iterations, most of the build time) and then optimizes it. Here the k-NN
// graph is the exact k-NN graph instead: inner products of a query batch against a tile
// of dataset rows with a rocBLAS SGEMM, then a top-k kernel keeps each row's nearest ids
// across the tiles. (The hipVS brute-force search computes the same distances several
// times slower; IVF-Flat k-NN was slower than this on every dataset tried.)
// The optimize step follows the CAGRA rules and runs as HIP kernels below:
//   1. prune: for node u and its rank-b neighbor B, count detours u->D->B where D is a
//      rank-d neighbor of u, B is a rank-x neighbor of D and max(d, x) < b; keep the
//      out_k neighbors with the fewest detours (ties by rank).
//   2. reverse edges: for each kept edge u->v (in rank order), record u in v's list.
//   3. merge: keep the first out_k/2 pruned edges, then add reverse edges, then fill
//      with the remaining pruned edges (no duplicates).
// Peak VRAM is kept close to the stock build: one resident copy of the vectors at a
// time, k-NN lists held in host memory until the optimize phase.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cuvs/core/c_api.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>

#define SK_MAX_K 256
#define SK_HASH 1024
#define SK_EMPTY 0xffffffffu

#ifndef SK_BATCH_ROWS
#define SK_BATCH_ROWS 2048     // query rows per exact k-NN step
#endif
#ifndef SK_CHUNK_ROWS
#define SK_CHUNK_ROWS 32768    // rows per prune launch (bounds the staging buffer)
#endif
#ifndef SK_GEMM_COLS
#define SK_GEMM_COLS 16384     // dataset rows per GEMM tile: workspace = batch x cols floats
#endif
#define SK_SEL_CAP 2048        // per-row candidate buffer of the tile top-k kernel (power of 2)
#define SK_SEL_THREADS 256

#ifdef SK_TIMING  // phase timings on stderr (diagnostics only)
#include <chrono>
static std::chrono::steady_clock::time_point& sk_t_last()
{
  static std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
  return t;
}
#define SK_T(label) do { hipDeviceSynchronize(); const auto t1 = std::chrono::steady_clock::now(); \
    fprintf(stderr, "[sk_time] %-14s %8.3f s\n", label, std::chrono::duration<double>(t1 - sk_t_last()).count()); \
    sk_t_last() = t1; } while (0)
#else
#define SK_T(label) do {} while (0)
#endif

namespace {

// Drop the row itself (and invalid ids) from m candidate lists of length kk, keep k.
__global__ void sk_pack_knn(const int64_t* nb, int64_t m, int64_t kk, int64_t row0, int64_t n,
                            uint32_t* out, int k)
{
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m) { return; }
  const int64_t* cand = nb + i * kk;
  uint32_t* o = out + i * k;
  int c = 0;
  for (int64_t j = 0; j < kk && c < k; ++j) {
    const int64_t v = cand[j];
    if (v == row0 + i || v < 0 || v >= n) { continue; }
    o[c++] = (uint32_t)v;
  }
  for (; c < k; ++c) { o[c] = o[c > 0 ? c - 1 : 0]; }
}

__global__ void sk_row_norms(const float* x, int64_t n, int64_t dim, float* out)
{
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const float* r = x + i * dim;
  float s = 0;
  for (int64_t j = 0; j < dim; ++j) { s += r[j] * r[j]; }
  out[i] = s;
}

// Ascending bitonic sort of the first p (a power of 2) (key, id) pairs in shared memory
// by the whole block.
__device__ void sk_bitonic(float* d, uint32_t* id, int p)
{
  for (int size = 2; size <= p; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      __syncthreads();
      for (int t = threadIdx.x; t < p / 2; t += blockDim.x) {
        const int i = 2 * t - (t & (stride - 1));
        const int j = i + stride;
        if ((d[i] > d[j]) == ((i & size) == 0)) {
          const float x = d[i]; d[i] = d[j]; d[j] = x;
          const uint32_t y = id[i]; id[i] = id[j]; id[j] = y;
        }
      }
    }
  }
  __syncthreads();
}

// One block per query row j of a GEMM tile: dot[j * cols + c] = <q_j, base[c0 + c]>.
// Keeps the kk smallest distances (norm[c] - 2 dot; the query norm is a per-row constant)
// across successive tiles in top_d / top_i, ascending. Candidates under the current kk-th
// distance are appended to a shared buffer that is sorted and cut back to kk when full.
__global__ void __launch_bounds__(SK_SEL_THREADS)
sk_tile_topk(const float* dot, const float* norm, int64_t c0, int64_t cols, int kk, bool first,
             float* top_d, int64_t* top_i)
{
  constexpr int SEG = SK_SEL_CAP / 2;  // columns scanned between buffer checks
  __shared__ float sd[SK_SEL_CAP];
  __shared__ uint32_t si[SK_SEL_CAP];
  __shared__ int cnt;
  const int64_t j = blockIdx.x;
  const float* row = dot + j * cols;
  float* td = top_d + j * kk;
  int64_t* ti = top_i + j * kk;
  for (int i = threadIdx.x; i < SK_SEL_CAP; i += blockDim.x) {
    const bool keep = !first && i < kk;
    sd[i] = keep ? td[i] : FLT_MAX;
    si[i] = keep ? (uint32_t)ti[i] : SK_EMPTY;
  }
  if (threadIdx.x == 0) { cnt = first ? 0 : kk; }
  __syncthreads();
  float thr = sd[kk - 1];
  int sorted = cnt;  // sd[0, sorted) is sorted; everything past cnt is FLT_MAX
  for (int64_t s = 0; s < cols; s += SEG) {
    const int64_t e = s + SEG < cols ? s + SEG : cols;
    for (int64_t c = s + threadIdx.x; c < e; c += blockDim.x) {
      const float d = norm[c0 + c] - 2.0f * row[c];
      if (d < thr) {
        const int p = atomicAdd(&cnt, 1);
        sd[p] = d; si[p] = (uint32_t)(c0 + c);
      }
    }
    __syncthreads();
    const int now = cnt;
    if (now != sorted && (now > SK_SEL_CAP - SEG || e == cols)) {
      int p = 2;
      while (p < now) { p <<= 1; }
      sk_bitonic(sd, si, p);
      for (int i = kk + threadIdx.x; i < p; i += blockDim.x) { sd[i] = FLT_MAX; si[i] = SK_EMPTY; }
      if (threadIdx.x == 0) { cnt = now < kk ? now : kk; }
      sorted = now < kk ? now : kk;
      __syncthreads();
      thr = sd[kk - 1];
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < kk; i += blockDim.x) {
    td[i] = sd[i];
    ti[i] = si[i] == SK_EMPTY ? -1 : (int64_t)si[i];
  }
}

__device__ __forceinline__ uint32_t sk_hash(uint32_t key) { return (key * 2654435761u) & (SK_HASH - 1); }

// Prunes rows [row0, row0 + m) of the k-NN graph; row u is written to pruned[u - row0].
__global__ void sk_prune(const uint32_t* knn, int64_t n, int64_t row0, int64_t m, int kin, int kout,
                         uint32_t* pruned)
{
  const int64_t u = row0 + blockIdx.x;
  if (blockIdx.x >= m || u >= n) { return; }
  __shared__ uint32_t ids[SK_MAX_K];
  __shared__ uint32_t det[SK_MAX_K];
  __shared__ uint32_t hkey[SK_HASH];
  __shared__ uint16_t hval[SK_HASH];
  const int t = threadIdx.x;
  for (int i = t; i < SK_HASH; i += blockDim.x) { hkey[i] = SK_EMPTY; }
  for (int i = t; i < kin; i += blockDim.x) { ids[i] = knn[u * kin + i]; det[i] = 0; }
  __syncthreads();
  // Rank lookup table: neighbor id -> its rank in u's list (first occurrence wins).
  if (t == 0) {
    for (int i = 0; i < kin; ++i) {
      uint32_t h = sk_hash(ids[i]);
      while (hkey[h] != SK_EMPTY && hkey[h] != ids[i]) { h = (h + 1) & (SK_HASH - 1); }
      if (hkey[h] == SK_EMPTY) { hkey[h] = ids[i]; hval[h] = (uint16_t)i; }
    }
  }
  __syncthreads();
  for (int d = 0; d < kin - 1; ++d) {
    const uint32_t* nd = knn + (int64_t)ids[d] * kin;
    for (int x = t; x < kin - 1; x += blockDim.x) {
      const uint32_t b_id = nd[x];
      uint32_t h = sk_hash(b_id);
      while (true) {
        const uint32_t key = hkey[h];
        if (key == SK_EMPTY) { break; }
        if (key == b_id) {
          const int b = hval[h];
          if (b > (d > x ? d : x)) { atomicAdd(&det[b], 1u); }
          break;
        }
        h = (h + 1) & (SK_HASH - 1);
      }
    }
  }
  __syncthreads();
  for (int i = t; i < kin; i += blockDim.x) {
    const uint64_t key_i = ((uint64_t)det[i] << 16) | (uint64_t)i;
    int pos = 0;
    for (int j = 0; j < kin; ++j) {
      const uint64_t key_j = ((uint64_t)det[j] << 16) | (uint64_t)j;
      pos += key_j < key_i;
    }
    if (pos < kout) { pruned[(u - row0) * kout + pos] = ids[i]; }
  }
}

__global__ void sk_reverse(const uint32_t* pruned, int64_t n, int kout, int rank, uint32_t* rev,
                           uint32_t* rev_cnt)
{
  const int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (u >= n) { return; }
  const uint32_t v = pruned[u * kout + rank];
  if (v >= n || v == u) { return; }
  const uint32_t pos = atomicAdd(&rev_cnt[v], 1u);
  if (pos < (uint32_t)kout) { rev[(int64_t)v * kout + pos] = (uint32_t)u; }
}

__device__ __forceinline__ bool sk_contains(const uint32_t* row, int len, uint32_t v)
{
  for (int i = 0; i < len; ++i) { if (row[i] == v) { return true; } }
  return false;
}

// Merges each row in place (one block per row): the pruned row is copied to shared memory
// and rewritten as the output. Shared rather than per-thread arrays keep the kernel free of
// private (scratch) memory, which the runtime would otherwise reserve for every wave.
__global__ void sk_merge(uint32_t* pruned, const uint32_t* rev, const uint32_t* rev_cnt,
                         int64_t n, int kout)
{
  const int64_t u = blockIdx.x;
  if (u >= n) { return; }
  __shared__ uint32_t p[SK_MAX_K];
  for (int i = threadIdx.x; i < kout; i += blockDim.x) { p[i] = pruned[u * kout + i]; }
  __syncthreads();
  if (threadIdx.x != 0) { return; }
  const uint32_t* r = rev + u * kout;
  uint32_t* o = pruned + u * kout;
  const int protect = kout / 2;
  int len = 0;
  for (int i = 0; i < protect; ++i) { if (!sk_contains(o, len, p[i])) { o[len++] = p[i]; } }
  const int nrev = rev_cnt[u] < (uint32_t)kout ? (int)rev_cnt[u] : kout;
  for (int i = 0; i < nrev && len < kout; ++i) { if (!sk_contains(o, len, r[i])) { o[len++] = r[i]; } }
  for (int i = protect; i < kout && len < kout; ++i) { if (!sk_contains(o, len, p[i])) { o[len++] = p[i]; } }
  for (int i = 0; len < kout; ++i) { o[len++] = p[i % kout]; }
}

// Device result buffers for one k-NN step of up to `rows` queries.
struct SkScratch {
  int64_t rows = 0, kk = 0;
  int64_t* r_id = nullptr;
  float* r_d = nullptr;
  bool alloc(int64_t r, int64_t k)
  {
    rows = r; kk = k;
    return hipMalloc(&r_id, sizeof(int64_t) * r * k) == hipSuccess &&
           hipMalloc(&r_d, sizeof(float) * r * k) == hipSuccess;
  }
  ~SkScratch() { (void)hipFree(r_id); (void)hipFree(r_d); }
};

// Exact k-NN over a device dataset: SGEMM tiles of SK_GEMM_COLS dataset rows against the
// query batch, each followed by the tile top-k kernel.
struct SkExact {
  rocblas_handle blas = nullptr;
  const float* base = nullptr;
  float* norm = nullptr;
  float* tile = nullptr;
  int64_t n = 0, cols = 0, rows = 0;
  ~SkExact() { close(); }
  void close()
  {
    if (blas) { rocblas_destroy_handle(blas); blas = nullptr; }
    (void)hipFree(norm); (void)hipFree(tile);
    norm = tile = nullptr;
  }
  int open(float* d_base, int64_t n_, int64_t dim, int64_t kk)
  {
    if (kk > SK_SEL_CAP / 2) { return 1; }
    base = d_base; n = n_;
    cols = std::min<int64_t>(n, SK_GEMM_COLS);
    rows = SK_BATCH_ROWS;
    if (rocblas_create_handle(&blas) != rocblas_status_success ||
        hipMalloc(&norm, sizeof(float) * n) != hipSuccess ||
        hipMalloc(&tile, sizeof(float) * rows * cols) != hipSuccess) { return 1; }
    hipLaunchKernelGGL(sk_row_norms, dim3((n + 255) / 256), dim3(256), 0, 0, d_base, n, dim, norm);
    return hipDeviceSynchronize() == hipSuccess ? 0 : 1;
  }
  // kk nearest rows for m <= rows queries; result left in sc.r_id / sc.r_d (ascending distance).
  int search(const float* d_q, int64_t m, int64_t dim, SkScratch& sc)
  {
    const float one = 1.0f, zero = 0.0f;
    for (int64_t c0 = 0; c0 < n; c0 += cols) {
      const int64_t w = std::min(cols, n - c0);
      // tile (w x m, column-major, ld w) = base[c0 .. c0 + w)^T * q
      if (rocblas_sgemm(blas, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int)w,
                        (rocblas_int)m, (rocblas_int)dim, &one, base + c0 * dim, (rocblas_int)dim, d_q,
                        (rocblas_int)dim, &zero, tile, (rocblas_int)w) != rocblas_status_success) { return 1; }
      hipLaunchKernelGGL(sk_tile_topk, dim3(m), dim3(SK_SEL_THREADS), 0, 0, tile, norm, c0, w,
                         (int)sc.kk, c0 == 0, sc.r_d, sc.r_id);
    }
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
  }
};

}  // namespace

// k-NN lists (n x kin, self excluded, ascending distance) of base into h_knn.
static int sk_knn_lists(const float* base, long n, long dim, int kin, std::vector<uint32_t>& h_knn)
{
  int rc = 0;
  const int64_t kk = kin + 1;
  const int64_t batch = n < SK_BATCH_ROWS ? n : SK_BATCH_ROWS;
  float* d_base = nullptr;
  uint32_t* d_pack = nullptr;
  SkExact exact;
  SkScratch sc;
#define SK_CHECK(x) do { if ((x) != hipSuccess) { rc = 2; goto done; } } while (0)
  SK_T("start");
  if (!sc.alloc(batch, kk)) { rc = 2; goto done; }
  SK_CHECK(hipMalloc(&d_base, sizeof(float) * n * dim));
  SK_CHECK(hipMemcpy(d_base, base, sizeof(float) * n * dim, hipMemcpyHostToDevice));
  if (exact.open(d_base, n, dim, kk) != 0) { rc = 3; goto done; }
  h_knn.resize((size_t)n * kin);
  SK_CHECK(hipMalloc(&d_pack, sizeof(uint32_t) * batch * kin));
  SK_T("knn_setup");
  for (int64_t s = 0; s < n; s += batch) {
    const int64_t m = n - s < batch ? n - s : batch;
    if (exact.search(d_base + s * dim, m, dim, sc) != 0) { rc = 4; goto done; }
    hipLaunchKernelGGL(sk_pack_knn, dim3((m + 255) / 256), dim3(256), 0, 0, sc.r_id, m, kk, s, (int64_t)n, d_pack, kin);
    SK_CHECK(hipGetLastError());
    SK_CHECK(hipMemcpy(h_knn.data() + s * kin, d_pack, sizeof(uint32_t) * m * kin, hipMemcpyDeviceToHost));
  }
  SK_T("knn_lists");
done:
#undef SK_CHECK
  exact.close();
  (void)hipFree(d_base); (void)hipFree(d_pack);
  return rc;
}

// CAGRA optimize (prune, reverse edges, merge) of host k-NN lists on the GPU. On success
// *graph_out is a device buffer (n x kout uint32, hipMalloc'ed) owned by the caller.
static int sk_optimize_graph(std::vector<uint32_t>& h_knn, long n, int kin, int kout,
                             unsigned int** graph_out)
{
  int rc = 0;
  const int64_t chunk = std::min<int64_t>(n, SK_CHUNK_ROWS);
  uint32_t *d_knn = nullptr, *d_chunk = nullptr, *d_pruned = nullptr, *d_rev = nullptr, *d_cnt = nullptr;
  std::vector<uint32_t> h_pruned;
#define SK_CHECK(x) do { if ((x) != hipSuccess) { rc = 2; goto done; } } while (0)
  // Prune: full k-NN lists on the device (2-hop lookups), pruned rows staged to host by
  // chunks, so the k-NN lists and the pruned graph are never resident together.
  SK_CHECK(hipMalloc(&d_knn, sizeof(uint32_t) * n * kin));
  SK_CHECK(hipMemcpy(d_knn, h_knn.data(), sizeof(uint32_t) * n * kin, hipMemcpyHostToDevice));
  std::vector<uint32_t>().swap(h_knn);
  SK_CHECK(hipMalloc(&d_chunk, sizeof(uint32_t) * chunk * kout));
  h_pruned.resize((size_t)n * kout);
  for (int64_t s = 0; s < n; s += chunk) {
    const int64_t m = std::min<int64_t>(chunk, n - s);
    hipLaunchKernelGGL(sk_prune, dim3(m), dim3(256), 0, 0, d_knn, (int64_t)n, s, m, kin, kout, d_chunk);
    SK_CHECK(hipGetLastError());
    SK_CHECK(hipMemcpy(h_pruned.data() + s * kout, d_chunk, sizeof(uint32_t) * m * kout, hipMemcpyDeviceToHost));
  }
  hipFree(d_chunk); d_chunk = nullptr;
  hipFree(d_knn); d_knn = nullptr;
  SK_T("prune");
  // Reverse edges + in-place merge: pruned graph + reverse lists.
  SK_CHECK(hipMalloc(&d_pruned, sizeof(uint32_t) * n * kout));
  SK_CHECK(hipMemcpy(d_pruned, h_pruned.data(), sizeof(uint32_t) * n * kout, hipMemcpyHostToDevice));
  std::vector<uint32_t>().swap(h_pruned);
  SK_CHECK(hipMalloc(&d_rev, sizeof(uint32_t) * n * kout));
  SK_CHECK(hipMalloc(&d_cnt, sizeof(uint32_t) * n));
  SK_CHECK(hipMemset(d_cnt, 0, sizeof(uint32_t) * n));
  for (int r = 0; r < kout; ++r) {
    hipLaunchKernelGGL(sk_reverse, dim3((n + 255) / 256), dim3(256), 0, 0, d_pruned, (int64_t)n, kout, r, d_rev, d_cnt);
  }
  SK_CHECK(hipGetLastError());
  hipLaunchKernelGGL(sk_merge, dim3(n), dim3(64), 0, 0, d_pruned, d_rev, d_cnt, (int64_t)n, kout);
  SK_CHECK(hipGetLastError());
  SK_CHECK(hipDeviceSynchronize());
  SK_T("reverse_merge");
  *graph_out = d_pruned;
  d_pruned = nullptr;
done:
#undef SK_CHECK
  hipFree(d_knn); hipFree(d_chunk); hipFree(d_pruned); hipFree(d_rev); hipFree(d_cnt);
  return rc;
}

// Builds a CAGRA graph for base (n x dim, host, row-major f32). On success *graph_out is
// a device buffer (n x kout uint32, hipMalloc'ed) owned by the caller.
// kin is the k-NN degree used for pruning (<= 256). Returns 0 on success.
// The k-NN workspace (GEMM tile, BLAS handle) is released before the optimize step, so it
// never overlaps the graph buffers or stays resident for the lifetime of the index.
extern "C" int sk_build_cagra_graph(cuvsResources_t, const float* base, long n, long dim,
                                    int kin, int kout, unsigned int** graph_out)
{
  *graph_out = nullptr;
  if (kin > SK_MAX_K || kout > kin || n <= kin + 1) { return 1; }
  std::vector<uint32_t> h_knn;
  int rc = sk_knn_lists(base, n, dim, kin, h_knn);
  if (rc == 0) { rc = sk_optimize_graph(h_knn, n, kin, kout, graph_out); }
  if (rc == 0) {
    fprintf(stderr, "[seekdb_cuvs] cagra graph: n=%ld dim=%ld knn=exact_gemm\n", n, dim);
  } else {
    fprintf(stderr, "[seekdb_cuvs] cagra graph build rc=%d\n", rc);
  }
  return rc;
}

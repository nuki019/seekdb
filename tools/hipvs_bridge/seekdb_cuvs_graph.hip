// CAGRA graph construction on the GPU without NN-descent.
//
// The stock CAGRA build creates an approximate k-NN graph with NN-descent (about 100
// refinement iterations, most of the build time) and then optimizes it. Here the k-NN
// graph is computed with hipVS search primitives instead:
//   * IVF-Flat k-NN when a row sample shows it reproduces the exact k-NN lists
//     (sampled list recall >= SK_IVF_MIN_RECALL with few probes), otherwise
//   * exact brute-force k-NN (tiled GPU distance computation, one pass).
// The optimize step follows the CAGRA rules and runs as HIP kernels below:
//   1. prune: for node u and its rank-b neighbor B, count detours u->D->B where D is a
//      rank-d neighbor of u, B is a rank-x neighbor of D and max(d, x) < b; keep the
//      out_k neighbors with the fewest detours (ties by rank).
//   2. reverse edges: for each kept edge u->v (in rank order), record u in v's list.
//   3. merge: keep the first out_k/2 pruned edges, then add reverse edges, then fill
//      with the remaining pruned edges (no duplicates).
#include <hip/hip_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <dlpack/dlpack.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>

#define SK_MAX_K 256
#define SK_HASH 1024
#define SK_EMPTY 0xffffffffu

namespace {

constexpr int64_t SK_BATCH = 16384;          // query rows per k-NN search call
constexpr long SK_IVF_MIN_ROWS = 50000;      // below this exact k-NN is cheap anyway
constexpr long SK_SAMPLE_ROWS = 2048;        // rows used to validate IVF k-NN quality
constexpr long SK_IVF_MAX_PROBES = 128;      // beyond this IVF is not cheaper than exact
constexpr double SK_IVF_MIN_RECALL = 0.96;   // sampled k-NN list recall required for IVF
constexpr double SK_IVF_GIVE_UP = 0.6;       // recall at 32 probes below this: use exact

__global__ void sk_pack_knn(const int64_t* nb, int64_t m, int64_t kk, int64_t row0, int64_t n,
                            uint32_t* knn, int k)
{
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m) { return; }
  const int64_t* cand = nb + i * kk;
  uint32_t* out = knn + (row0 + i) * k;
  int c = 0;
  for (int64_t j = 0; j < kk && c < k; ++j) {
    const int64_t v = cand[j];
    if (v == row0 + i || v < 0 || v >= n) { continue; }
    out[c++] = (uint32_t)v;
  }
  for (; c < k; ++c) { out[c] = out[c > 0 ? c - 1 : 0]; }
}

__device__ __forceinline__ uint32_t sk_hash(uint32_t key) { return (key * 2654435761u) & (SK_HASH - 1); }

__global__ void sk_prune(const uint32_t* knn, int64_t n, int kin, int kout, uint32_t* pruned)
{
  const int64_t u = blockIdx.x;
  if (u >= n) { return; }
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
    if (pos < kout) { pruned[u * kout + pos] = ids[i]; }
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

__global__ void sk_merge(const uint32_t* pruned, const uint32_t* rev, const uint32_t* rev_cnt,
                         int64_t n, int kout, uint32_t* out)
{
  const int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (u >= n) { return; }
  const uint32_t* p = pruned + u * kout;
  const uint32_t* r = rev + u * kout;
  uint32_t* o = out + u * kout;
  const int protect = kout / 2;
  int len = 0;
  for (int i = 0; i < protect; ++i) { if (!sk_contains(o, len, p[i])) { o[len++] = p[i]; } }
  const int nrev = rev_cnt[u] < (uint32_t)kout ? (int)rev_cnt[u] : kout;
  for (int i = 0; i < nrev && len < kout; ++i) { if (!sk_contains(o, len, r[i])) { o[len++] = r[i]; } }
  for (int i = protect; i < kout && len < kout; ++i) { if (!sk_contains(o, len, p[i])) { o[len++] = p[i]; } }
  for (int i = 0; len < kout; ++i) { o[len++] = p[i % kout]; }
}

void sk_tensor(DLManagedTensor* t, void* data, int64_t* shape, uint8_t code, uint8_t bits)
{
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCUDA; t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = code; t->dl_tensor.dtype.bits = bits; t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape; t->dl_tensor.strides = nullptr; t->dl_tensor.byte_offset = 0;
  t->manager_ctx = nullptr; t->deleter = nullptr;
}

// k-NN (kk neighbors incl. self) for the m rows of d_q with IVF-Flat when ivf is set,
// brute force otherwise. Neighbors stay in d_nb; copied to h_nb when it is not null.
int sk_search(cuvsResources_t res, cuvsBruteForceIndex_t bf, cuvsIvfFlatIndex_t ivf,
              cuvsIvfFlatSearchParams_t sp, const float* d_q, int64_t m, int64_t dim, int64_t kk,
              int64_t* d_nb, float* d_ds, int64_t* h_nb)
{
  int64_t qs[2] = {m, dim}, os[2] = {m, kk};
  DLManagedTensor qt, nt, ot;
  sk_tensor(&qt, const_cast<float*>(d_q), qs, kDLFloat, 32);
  sk_tensor(&nt, d_nb, os, kDLInt, 64);
  sk_tensor(&ot, d_ds, os, kDLFloat, 32);
  cuvsFilter flt; flt.type = NO_FILTER; flt.addr = 0;
  const cuvsError_t sr = ivf ? cuvsIvfFlatSearch(res, sp, ivf, &qt, &nt, &ot, flt)
                             : cuvsBruteForceSearch(res, bf, &qt, &nt, &ot, flt);
  if (sr != CUVS_SUCCESS || cuvsStreamSync(res) != CUVS_SUCCESS) { return 1; }
  if (h_nb && hipMemcpy(h_nb, d_nb, sizeof(int64_t) * m * kk, hipMemcpyDeviceToHost) != hipSuccess) { return 2; }
  return 0;
}

// Sorted neighbor list of one row without the row itself (at most k entries).
int sk_row_set(const int64_t* cand, int64_t kk, int64_t self, int k, int64_t* out)
{
  int c = 0;
  for (int64_t j = 0; j < kk && c < k; ++j) { if (cand[j] != self && cand[j] >= 0) { out[c++] = cand[j]; } }
  std::sort(out, out + c);
  return c;
}

// Smallest IVF probe count whose k-NN lists match the exact lists on a row sample with
// recall >= SK_IVF_MIN_RECALL, or -1 if none does (then exact k-NN is used).
long sk_pick_probes(cuvsResources_t res, cuvsBruteForceIndex_t bf, cuvsIvfFlatIndex_t ivf,
                    cuvsIvfFlatSearchParams_t sp, const float* base, int64_t n, int64_t dim,
                    int kin, long lists, double* est)
{
  const int64_t m = n < SK_SAMPLE_ROWS ? n : SK_SAMPLE_ROWS, kk = kin + 1, step = n / m;
  std::vector<float> h_q(m * dim);
  std::vector<int64_t> rows(m), exact(m * kk), approx(m * kk), a(kin), b(kin);
  for (int64_t i = 0; i < m; ++i) {
    rows[i] = i * step;
    memcpy(h_q.data() + i * dim, base + rows[i] * dim, sizeof(float) * dim);
  }
  float* d_q = nullptr; int64_t* d_nb = nullptr; float* d_ds = nullptr;
  long picked = -1;
  *est = 0;
  if (hipMalloc(&d_q, sizeof(float) * m * dim) == hipSuccess &&
      hipMalloc(&d_nb, sizeof(int64_t) * m * kk) == hipSuccess &&
      hipMalloc(&d_ds, sizeof(float) * m * kk) == hipSuccess &&
      hipMemcpy(d_q, h_q.data(), sizeof(float) * m * dim, hipMemcpyHostToDevice) == hipSuccess &&
      sk_search(res, bf, nullptr, nullptr, d_q, m, dim, kk, d_nb, d_ds, exact.data()) == 0) {
    const long max_probes = std::min<long>(SK_IVF_MAX_PROBES, lists / 2);
    for (long p = 16; p <= max_probes && picked < 0; p *= 2) {
      sp->n_probes = (uint32_t)p;
      if (sk_search(res, nullptr, ivf, sp, d_q, m, dim, kk, d_nb, d_ds, approx.data()) != 0) { break; }
      int64_t hit = 0, total = 0;
      for (int64_t i = 0; i < m; ++i) {
        const int ca = sk_row_set(exact.data() + i * kk, kk, rows[i], kin, a.data());
        const int cb = sk_row_set(approx.data() + i * kk, kk, rows[i], kin, b.data());
        total += ca;
        for (int x = 0, y = 0; x < ca && y < cb;) {
          if (a[x] == b[y]) { ++hit; ++x; ++y; } else if (a[x] < b[y]) { ++x; } else { ++y; }
        }
      }
      *est = total ? (double)hit / total : 0;
      if (*est >= SK_IVF_MIN_RECALL) { picked = p; }
      else if (p >= 32 && *est < SK_IVF_GIVE_UP) { break; }
    }
  }
  hipFree(d_q); hipFree(d_nb); hipFree(d_ds);
  return picked;
}

}  // namespace

// Builds a CAGRA graph (n x kout, host) for base (n x dim, host, row-major f32).
// kin is the k-NN degree used for pruning (<= 256). Returns 0 on success.
extern "C" int sk_build_cagra_graph(cuvsResources_t res, const float* base, long n, long dim,
                                    int kin, int kout, unsigned int* out_graph)
{
  if (kin > SK_MAX_K || kout > kin || n <= kin + 1) { return 1; }
  int rc = 0;
  const int64_t kk = kin + 1;
  const int64_t batch = n < SK_BATCH ? n : SK_BATCH;
  float* d_base = nullptr; int64_t* d_nb = nullptr; float* d_ds = nullptr;
  uint32_t *d_knn = nullptr, *d_pruned = nullptr, *d_rev = nullptr, *d_cnt = nullptr, *d_out = nullptr;
  cuvsBruteForceIndex_t idx = nullptr;
  cuvsIvfFlatIndex_t ivf_idx = nullptr;
  cuvsIvfFlatIndexParams_t ivf_ip = nullptr;
  cuvsIvfFlatSearchParams_t ivf_sp = nullptr;
  long ivf_lists = 0, ivf_probes = -1;
  double est = 0;
#define SK_CHECK(x) do { if ((x) != hipSuccess) { rc = 2; goto done; } } while (0)
  SK_CHECK(hipMalloc(&d_base, sizeof(float) * n * dim));
  SK_CHECK(hipMalloc(&d_nb, sizeof(int64_t) * batch * kk));
  SK_CHECK(hipMalloc(&d_ds, sizeof(float) * batch * kk));
  SK_CHECK(hipMalloc(&d_knn, sizeof(uint32_t) * n * kin));
  SK_CHECK(hipMemcpy(d_base, base, sizeof(float) * n * dim, hipMemcpyHostToDevice));
  {
    int64_t bs[2] = {n, dim};
    DLManagedTensor dt; sk_tensor(&dt, d_base, bs, kDLFloat, 32);
    if (cuvsBruteForceIndexCreate(&idx) != CUVS_SUCCESS ||
        cuvsBruteForceBuild(res, &dt, L2Expanded, 0.0f, idx) != CUVS_SUCCESS) { rc = 3; goto done; }
    if (n >= SK_IVF_MIN_ROWS) {
      ivf_lists = (long)(2.0 * sqrt((double)n));
      if (cuvsIvfFlatIndexParamsCreate(&ivf_ip) != CUVS_SUCCESS ||
          cuvsIvfFlatSearchParamsCreate(&ivf_sp) != CUVS_SUCCESS ||
          cuvsIvfFlatIndexCreate(&ivf_idx) != CUVS_SUCCESS) { rc = 3; goto done; }
      ivf_ip->n_lists = (uint32_t)ivf_lists;
      ivf_ip->metric = L2Expanded;
      if (cuvsIvfFlatBuild(res, ivf_ip, &dt, ivf_idx) != CUVS_SUCCESS) { rc = 3; goto done; }
      ivf_probes = sk_pick_probes(res, idx, ivf_idx, ivf_sp, base, n, dim, kin, ivf_lists, &est);
      if (ivf_probes <= 0) {
        cuvsIvfFlatIndexDestroy(ivf_idx); ivf_idx = nullptr;
      } else {
        ivf_sp->n_probes = (uint32_t)ivf_probes;
      }
    }
  }
  for (int64_t s = 0; s < n; s += batch) {
    const int64_t m = n - s < batch ? n - s : batch;
    if (sk_search(res, idx, ivf_idx, ivf_sp, d_base + s * dim, m, dim, kk, d_nb, d_ds, nullptr) != 0) { rc = 4; goto done; }
    hipLaunchKernelGGL(sk_pack_knn, dim3((m + 255) / 256), dim3(256), 0, 0, d_nb, m, kk, s, (int64_t)n, d_knn, kin);
    SK_CHECK(hipDeviceSynchronize());
  }
  hipFree(d_nb); d_nb = nullptr;
  hipFree(d_ds); d_ds = nullptr;
  hipFree(d_base); d_base = nullptr;
  SK_CHECK(hipMalloc(&d_pruned, sizeof(uint32_t) * n * kout));
  SK_CHECK(hipMalloc(&d_rev, sizeof(uint32_t) * n * kout));
  SK_CHECK(hipMalloc(&d_cnt, sizeof(uint32_t) * n));
  SK_CHECK(hipMemset(d_cnt, 0, sizeof(uint32_t) * n));
  hipLaunchKernelGGL(sk_prune, dim3(n), dim3(256), 0, 0, d_knn, (int64_t)n, kin, kout, d_pruned);
  SK_CHECK(hipGetLastError());
  SK_CHECK(hipDeviceSynchronize());
  hipFree(d_knn); d_knn = nullptr;
  for (int r = 0; r < kout; ++r) {
    hipLaunchKernelGGL(sk_reverse, dim3((n + 255) / 256), dim3(256), 0, 0, d_pruned, (int64_t)n, kout, r, d_rev, d_cnt);
  }
  SK_CHECK(hipGetLastError());
  SK_CHECK(hipMalloc(&d_out, sizeof(uint32_t) * n * kout));
  hipLaunchKernelGGL(sk_merge, dim3((n + 127) / 128), dim3(128), 0, 0, d_pruned, d_rev, d_cnt, (int64_t)n, kout, d_out);
  SK_CHECK(hipGetLastError());
  SK_CHECK(hipMemcpy(out_graph, d_out, sizeof(uint32_t) * n * kout, hipMemcpyDeviceToHost));
  fprintf(stderr, "[seekdb_cuvs] cagra graph: n=%ld dim=%ld knn=%s probes=%ld sampled_recall=%.4f\n",
          n, dim, ivf_idx ? "ivf_flat" : "exact", ivf_idx ? ivf_probes : 0L, est);
done:
#undef SK_CHECK
  if (idx) { cuvsBruteForceIndexDestroy(idx); }
  if (ivf_idx) { cuvsIvfFlatIndexDestroy(ivf_idx); }
  if (ivf_ip) { cuvsIvfFlatIndexParamsDestroy(ivf_ip); }
  if (ivf_sp) { cuvsIvfFlatSearchParamsDestroy(ivf_sp); }
  hipFree(d_base); hipFree(d_nb); hipFree(d_ds); hipFree(d_knn);
  hipFree(d_pruned); hipFree(d_rev); hipFree(d_cnt); hipFree(d_out);
  if (rc) { fprintf(stderr, "[seekdb_cuvs] cagra graph build rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  return rc;
}

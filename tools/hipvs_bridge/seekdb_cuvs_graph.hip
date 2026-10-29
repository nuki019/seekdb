// CAGRA graph construction on the GPU without NN-descent.
//
// The stock CAGRA build creates an approximate k-NN graph with NN-descent (about 100
// refinement iterations, most of the build time) and then optimizes it. Here the k-NN
// graph is computed with hipVS search primitives instead:
//   * IVF-Flat k-NN when a row sample shows it reproduces the exact k-NN lists
//     (sampled list recall >= SK_IVF_MIN_RECALL with few probes), otherwise
//   * exact brute-force k-NN, tiled over row chunks of the dataset so the search
//     workspace stays bounded, with the per-chunk top-k merged on the GPU.
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
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <dlpack/dlpack.h>
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
#define SK_BATCH_ROWS 2048     // query rows per brute-force k-NN step
#endif
#ifndef SK_CHUNK_ROWS
#define SK_CHUNK_ROWS 32768    // dataset rows per brute-force tile: workspace ~ batch x chunk floats
#endif
#ifndef SK_USE_IVF
#define SK_USE_IVF 1          // IVF-Flat k-NN for clustered data (validated per dataset)
#endif
#ifndef SK_IVF_BATCH_ROWS
#define SK_IVF_BATCH_ROWS 1024 // query rows per IVF-Flat k-NN step (bounds its search workspace)
#endif

namespace {

constexpr long SK_IVF_MIN_ROWS = 50000;      // below this exact k-NN is cheap anyway
constexpr long SK_SAMPLE_ROWS = 2048;        // rows used to validate IVF k-NN quality
constexpr int64_t SK_SAMPLE_STEP_ROWS = 512; // sample rows per search call (bounds workspace)
constexpr int64_t SK_GATE_ROWS = 65536;      // subset used for the cheap IVF pre-check
constexpr int64_t SK_GATE_SAMPLE_ROWS = 512; // sample rows of that subset
constexpr double SK_GATE_MIN_RECALL = 0.9;   // pre-check bar (the full check still applies)
constexpr long SK_IVF_MAX_PROBES = 128;      // beyond this IVF is not cheaper than exact
constexpr double SK_IVF_MIN_RECALL = 0.96;   // sampled k-NN list recall required for IVF
constexpr double SK_IVF_GIVE_UP = 0.6;       // recall at 32 probes below this: use exact

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

// Merge two ascending top-kk lists per row (b's ids shifted by b_off) into the best kk.
__global__ void sk_topk_merge(const int64_t* a_id, const float* a_d, const int64_t* b_id,
                              const float* b_d, int64_t b_off, int64_t m, int kk, bool first,
                              int64_t* o_id, float* o_d)
{
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= m) { return; }
  const int64_t* ai = a_id + i * kk; const float* ad = a_d + i * kk;
  const int64_t* bi = b_id + i * kk; const float* bd = b_d + i * kk;
  int64_t* oi = o_id + i * kk; float* od = o_d + i * kk;
  int x = 0, y = 0;
  for (int c = 0; c < kk; ++c) {
    const float da = (!first && x < kk) ? ad[x] : FLT_MAX;
    const float db = (y < kk && bi[y] >= 0) ? bd[y] : FLT_MAX;
    if (da <= db && !first && x < kk) { oi[c] = ai[x]; od[c] = da; ++x; }
    else if (y < kk) { oi[c] = bi[y] >= 0 ? bi[y] + b_off : -1; od[c] = db; ++y; }
    else { oi[c] = -1; od[c] = FLT_MAX; }
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

void sk_tensor(DLManagedTensor* t, void* data, int64_t* shape, uint8_t code, uint8_t bits)
{
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCUDA; t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = code; t->dl_tensor.dtype.bits = bits; t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape; t->dl_tensor.strides = nullptr; t->dl_tensor.byte_offset = 0;
  t->manager_ctx = nullptr; t->deleter = nullptr;
}

// Device scratch for one k-NN step of up to `rows` queries.
struct SkScratch {
  int64_t rows = 0, kk = 0;
  int64_t *t_id = nullptr, *r_id = nullptr, *o_id = nullptr;  // tile result, running, output
  float *t_d = nullptr, *r_d = nullptr, *o_d = nullptr;
  bool alloc(int64_t r, int64_t k)
  {
    rows = r; kk = k;
    const size_t ni = sizeof(int64_t) * r * k, nf = sizeof(float) * r * k;
    return hipMalloc(&t_id, ni) == hipSuccess && hipMalloc(&r_id, ni) == hipSuccess &&
           hipMalloc(&o_id, ni) == hipSuccess && hipMalloc(&t_d, nf) == hipSuccess &&
           hipMalloc(&r_d, nf) == hipSuccess && hipMalloc(&o_d, nf) == hipSuccess;
  }
  ~SkScratch() { hipFree(t_id); hipFree(r_id); hipFree(o_id); hipFree(t_d); hipFree(r_d); hipFree(o_d); }
};

// Exact k-NN over a device dataset, tiled into row chunks (each a brute-force view).
struct SkBruteForce {
  std::vector<cuvsBruteForceIndex_t> idx;
  std::vector<int64_t> off;
  ~SkBruteForce() { close(); }
  void close()
  {
    for (auto i : idx) { if (i) { cuvsBruteForceIndexDestroy(i); } }
    idx.clear(); off.clear();
  }
  int open(cuvsResources_t res, float* d_base, int64_t n, int64_t dim, int64_t kk)
  {
    const int64_t chunk = std::max<int64_t>(SK_CHUNK_ROWS, kk);
    for (int64_t s = 0; s < n;) {
      int64_t r = std::min(chunk, n - s);
      if (n - (s + r) > 0 && n - (s + r) < kk) { r = n - s; }  // a tail shorter than kk joins this chunk
      int64_t bs[2] = {r, dim};
      DLManagedTensor dt; sk_tensor(&dt, d_base + s * dim, bs, kDLFloat, 32);
      cuvsBruteForceIndex_t i = nullptr;
      if (cuvsBruteForceIndexCreate(&i) != CUVS_SUCCESS) { return 1; }
      idx.push_back(i); off.push_back(s);
      if (cuvsBruteForceBuild(res, &dt, L2Expanded, 0.0f, i) != CUVS_SUCCESS) { return 1; }
      s += r;
    }
    return 0;
  }
  // kk nearest rows for m queries; result left in sc.r_id / sc.r_d (ascending distance).
  int search(cuvsResources_t res, const float* d_q, int64_t m, int64_t dim, SkScratch& sc)
  {
    int64_t qs[2] = {m, dim}, os[2] = {m, sc.kk};
    DLManagedTensor qt, nt, ot;
    sk_tensor(&qt, const_cast<float*>(d_q), qs, kDLFloat, 32);
    sk_tensor(&nt, sc.t_id, os, kDLInt, 64);
    sk_tensor(&ot, sc.t_d, os, kDLFloat, 32);
    cuvsFilter flt; flt.type = NO_FILTER; flt.addr = 0;
    for (size_t c = 0; c < idx.size(); ++c) {
      if (cuvsBruteForceSearch(res, idx[c], &qt, &nt, &ot, flt) != CUVS_SUCCESS ||
          cuvsStreamSync(res) != CUVS_SUCCESS) { return 1; }
      hipLaunchKernelGGL(sk_topk_merge, dim3((m + 255) / 256), dim3(256), 0, 0, sc.r_id, sc.r_d,
                         sc.t_id, sc.t_d, off[c], m, (int)sc.kk, c == 0, sc.o_id, sc.o_d);
      if (hipDeviceSynchronize() != hipSuccess) { return 2; }
      std::swap(sc.r_id, sc.o_id); std::swap(sc.r_d, sc.o_d);
    }
    return 0;
  }
};

// IVF-Flat k-NN for m queries; result left in sc.r_id / sc.r_d.
int sk_ivf_search(cuvsResources_t res, cuvsIvfFlatIndex_t ivf, cuvsIvfFlatSearchParams_t sp,
                  const float* d_q, int64_t m, int64_t dim, SkScratch& sc)
{
  int64_t qs[2] = {m, dim}, os[2] = {m, sc.kk};
  DLManagedTensor qt, nt, ot;
  sk_tensor(&qt, const_cast<float*>(d_q), qs, kDLFloat, 32);
  sk_tensor(&nt, sc.r_id, os, kDLInt, 64);
  sk_tensor(&ot, sc.r_d, os, kDLFloat, 32);
  cuvsFilter flt; flt.type = NO_FILTER; flt.addr = 0;
  if (cuvsIvfFlatSearch(res, sp, ivf, &qt, &nt, &ot, flt) != CUVS_SUCCESS ||
      cuvsStreamSync(res) != CUVS_SUCCESS) { return 1; }
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

// Evenly spaced row sample and its exact k-NN lists, used to validate IVF k-NN quality.
struct SkSample {
  int64_t m = 0, kk = 0, dim = 0;
  std::vector<float> q;         // m x dim query rows
  std::vector<int64_t> rows;    // their row ids
  std::vector<int64_t> exact;   // m x kk exact neighbors (incl. self)
  float* d_q = nullptr;
  ~SkSample() { hipFree(d_q); }
};

int sk_sample_exact(cuvsResources_t res, SkBruteForce& bf, const float* base, int64_t n,
                    int64_t dim, int kin, SkScratch& sc, SkSample& s)
{
  s.m = std::min<int64_t>(std::min<int64_t>(n, SK_SAMPLE_ROWS), sc.rows);
  s.kk = kin + 1;
  s.dim = dim;
  const int64_t step = n / s.m;
  s.q.resize(s.m * dim);
  s.rows.resize(s.m);
  s.exact.resize(s.m * s.kk);
  for (int64_t i = 0; i < s.m; ++i) {
    s.rows[i] = i * step;
    memcpy(s.q.data() + i * dim, base + s.rows[i] * dim, sizeof(float) * dim);
  }
  if (hipMalloc(&s.d_q, sizeof(float) * s.m * dim) != hipSuccess ||
      hipMemcpy(s.d_q, s.q.data(), sizeof(float) * s.m * dim, hipMemcpyHostToDevice) != hipSuccess) {
    return 1;
  }
  for (int64_t b = 0; b < s.m; b += SK_SAMPLE_STEP_ROWS) {
    const int64_t m = std::min<int64_t>(SK_SAMPLE_STEP_ROWS, s.m - b);
    if (bf.search(res, s.d_q + b * dim, m, dim, sc) != 0 ||
        hipMemcpy(s.exact.data() + b * s.kk, sc.r_id, sizeof(int64_t) * m * s.kk, hipMemcpyDeviceToHost) != hipSuccess) {
      return 1;
    }
  }
  return 0;
}

// Smallest IVF probe count (16, 32, ... <= max_probes) whose k-NN lists match the
// sample's exact lists with recall >= min_recall, or -1 if none does.
long sk_pick_probes(cuvsResources_t res, cuvsIvfFlatIndex_t ivf, cuvsIvfFlatSearchParams_t sp,
                    const SkSample& s, int kin, long max_probes, double min_recall, SkScratch& sc,
                    double* est)
{
  long picked = -1;
  *est = 0;
  std::vector<int64_t> approx(s.m * s.kk), a(kin), b(kin);
  for (long p = 16; p <= max_probes && picked < 0; p *= 2) {
    sp->n_probes = (uint32_t)p;
    bool ok = true;
    for (int64_t b = 0; b < s.m && ok; b += SK_SAMPLE_STEP_ROWS) {
      const int64_t m = std::min<int64_t>(SK_SAMPLE_STEP_ROWS, s.m - b);
      ok = sk_ivf_search(res, ivf, sp, s.d_q + b * s.dim, m, s.dim, sc) == 0 &&
           hipMemcpy(approx.data() + b * s.kk, sc.r_id, sizeof(int64_t) * m * s.kk, hipMemcpyDeviceToHost) == hipSuccess;
    }
    if (!ok) { break; }
    int64_t hit = 0, total = 0;
    for (int64_t i = 0; i < s.m; ++i) {
      const int ca = sk_row_set(s.exact.data() + i * s.kk, s.kk, s.rows[i], kin, a.data());
      const int cb = sk_row_set(approx.data() + i * s.kk, s.kk, s.rows[i], kin, b.data());
      total += ca;
      for (int x = 0, y = 0; x < ca && y < cb;) {
        if (a[x] == b[y]) { ++hit; ++x; ++y; } else if (a[x] < b[y]) { ++x; } else { ++y; }
      }
    }
    *est = total ? (double)hit / total : 0;
    if (*est >= min_recall) { picked = p; }
    else if (p >= 32 && *est < SK_IVF_GIVE_UP) { break; }
  }
  return picked;
}

// IVF-Flat index over the device dataset d (n x dim) with ~2 sqrt(n) lists.
int sk_ivf_open(cuvsResources_t res, float* d, int64_t n, int64_t dim, double train_fraction,
                cuvsIvfFlatIndexParams_t* ip, cuvsIvfFlatSearchParams_t* sp, cuvsIvfFlatIndex_t* idx,
                long* lists)
{
  *lists = (long)(2.0 * sqrt((double)n));
  if (cuvsIvfFlatIndexParamsCreate(ip) != CUVS_SUCCESS ||
      cuvsIvfFlatSearchParamsCreate(sp) != CUVS_SUCCESS ||
      cuvsIvfFlatIndexCreate(idx) != CUVS_SUCCESS) { return 1; }
  (*ip)->n_lists = (uint32_t)*lists;
  (*ip)->metric = L2Expanded;
  (*ip)->kmeans_trainset_fraction = train_fraction;
  int64_t bs[2] = {n, dim};
  DLManagedTensor dt; sk_tensor(&dt, d, bs, kDLFloat, 32);
  return cuvsIvfFlatBuild(res, *ip, &dt, *idx) == CUVS_SUCCESS && cuvsStreamSync(res) == CUVS_SUCCESS ? 0 : 1;
}

// Cheap pre-check on an evenly spaced subset of SK_GATE_ROWS rows (its own small IVF
// index and exact sample): datasets on which IVF-Flat clearly cannot reproduce exact
// k-NN lists go straight to exact k-NN, without building the full IVF index.
bool sk_ivf_promising(cuvsResources_t res, const float* base, int64_t n, int64_t dim, int kin,
                      SkScratch& sc)
{
  const int64_t ns = std::min<int64_t>(n, SK_GATE_ROWS), step = n / ns, kk = kin + 1;
  std::vector<float> h_sub(ns * dim);
  for (int64_t i = 0; i < ns; ++i) { memcpy(h_sub.data() + i * dim, base + i * step * dim, sizeof(float) * dim); }
  float* d_sub = nullptr;
  SkBruteForce bf;
  SkSample sample;
  cuvsIvfFlatIndexParams_t ip = nullptr;
  cuvsIvfFlatSearchParams_t sp = nullptr;
  cuvsIvfFlatIndex_t ivf = nullptr;
  long lists = 0;
  double est = 0;
  bool promising = true;  // on any failure, let the full check decide
  if (hipMalloc(&d_sub, sizeof(float) * ns * dim) == hipSuccess &&
      hipMemcpy(d_sub, h_sub.data(), sizeof(float) * ns * dim, hipMemcpyHostToDevice) == hipSuccess &&
      bf.open(res, d_sub, ns, dim, kk) == 0) {
    // Sample of the subset with exact lists inside the subset.
    sample.m = std::min<int64_t>(ns, SK_GATE_SAMPLE_ROWS);
    sample.kk = kk;
    sample.dim = dim;
    const int64_t sstep = ns / sample.m;
    sample.q.resize(sample.m * dim);
    sample.rows.resize(sample.m);
    sample.exact.resize(sample.m * kk);
    for (int64_t i = 0; i < sample.m; ++i) {
      sample.rows[i] = i * sstep;
      memcpy(sample.q.data() + i * dim, h_sub.data() + sample.rows[i] * dim, sizeof(float) * dim);
    }
    bool ok = hipMalloc(&sample.d_q, sizeof(float) * sample.m * dim) == hipSuccess &&
              hipMemcpy(sample.d_q, sample.q.data(), sizeof(float) * sample.m * dim, hipMemcpyHostToDevice) == hipSuccess &&
              bf.search(res, sample.d_q, sample.m, dim, sc) == 0 &&
              hipMemcpy(sample.exact.data(), sc.r_id, sizeof(int64_t) * sample.m * kk, hipMemcpyDeviceToHost) == hipSuccess;
    bf.close();
    if (ok && sk_ivf_open(res, d_sub, ns, dim, 0.5, &ip, &sp, &ivf, &lists) == 0) {
      // Same fraction of the lists as the full check probes at most.
      const long max_probes = std::max<long>(16, (long)(SK_IVF_MAX_PROBES * (double)lists / (2.0 * sqrt((double)n))));
      promising = sk_pick_probes(res, ivf, sp, sample, kin, max_probes, SK_GATE_MIN_RECALL, sc, &est) > 0;
    }
  }
  if (ivf) { cuvsIvfFlatIndexDestroy(ivf); }
  if (ip) { cuvsIvfFlatIndexParamsDestroy(ip); }
  if (sp) { cuvsIvfFlatSearchParamsDestroy(sp); }
  bf.close();
  hipFree(d_sub);
  return promising;
}

}  // namespace

// k-NN lists (n x kin, self excluded, ascending distance) of base into h_knn, using the
// hipVS search primitives on `res`. *probes > 0 reports that IVF-Flat was used.
static int sk_knn_lists(cuvsResources_t res, const float* base, long n, long dim, int kin,
                        std::vector<uint32_t>& h_knn, long* probes, double* sampled_recall)
{
  int rc = 0;
  const int64_t kk = kin + 1;
  const int64_t batch = n < SK_BATCH_ROWS ? n : SK_BATCH_ROWS;
  float* d_base = nullptr; float* d_q = nullptr;
  uint32_t* d_pack = nullptr;
  cuvsIvfFlatIndex_t ivf_idx = nullptr;
  cuvsIvfFlatIndexParams_t ivf_ip = nullptr;
  cuvsIvfFlatSearchParams_t ivf_sp = nullptr;
  long ivf_lists = 0, ivf_probes = -1;
  double est = 0;
  SkBruteForce bf;
  SkScratch sc;
#define SK_CHECK(x) do { if ((x) != hipSuccess) { rc = 2; goto done; } } while (0)
  if (!sc.alloc(batch, kk)) { rc = 2; goto done; }
  {
    // Phase 0: cheap subset pre-check; hard datasets skip the IVF attempt entirely.
    SkSample sample;
    bool try_ivf = SK_USE_IVF && n >= SK_IVF_MIN_ROWS && sk_ivf_promising(res, base, n, dim, kin, sc);
    // Phase 1: exact k-NN (of a row sample when IVF is still a candidate) over the device dataset.
    SK_CHECK(hipMalloc(&d_base, sizeof(float) * n * dim));
    SK_CHECK(hipMemcpy(d_base, base, sizeof(float) * n * dim, hipMemcpyHostToDevice));
    if (bf.open(res, d_base, n, dim, kk) != 0) { rc = 3; goto done; }
    if (try_ivf && sk_sample_exact(res, bf, base, n, dim, kin, sc, sample) != 0) { try_ivf = false; }
    if (try_ivf) {
      // Phase 2: IVF-Flat index (its own copy of the vectors), validated on the sample.
      bf.close();
      const bool built = sk_ivf_open(res, d_base, n, dim, 0.1, &ivf_ip, &ivf_sp, &ivf_idx, &ivf_lists) == 0;
      hipFree(d_base); d_base = nullptr;
      if (built) {
        ivf_probes = sk_pick_probes(res, ivf_idx, ivf_sp, sample, kin,
                                    std::min<long>(SK_IVF_MAX_PROBES, ivf_lists / 2), SK_IVF_MIN_RECALL, sc, &est);
      }
      if (ivf_probes > 0) {
        ivf_sp->n_probes = (uint32_t)ivf_probes;
        SK_CHECK(hipMalloc(&d_q, sizeof(float) * batch * dim));  // query batches come from host
      } else {
        cuvsIvfFlatIndexDestroy(ivf_idx); ivf_idx = nullptr;
        SK_CHECK(hipMalloc(&d_base, sizeof(float) * n * dim));
        SK_CHECK(hipMemcpy(d_base, base, sizeof(float) * n * dim, hipMemcpyHostToDevice));
        if (bf.open(res, d_base, n, dim, kk) != 0) { rc = 3; goto done; }
      }
    }
  }
  // Phase 3: k-NN lists of all rows, collected in host memory.
  h_knn.resize((size_t)n * kin);
  SK_CHECK(hipMalloc(&d_pack, sizeof(uint32_t) * batch * kin));
  for (int64_t s = 0, step = ivf_idx ? std::min<int64_t>(batch, SK_IVF_BATCH_ROWS) : batch; s < n; s += step) {
    const int64_t m = n - s < step ? n - s : step;
    if (ivf_idx) {
      SK_CHECK(hipMemcpy(d_q, base + s * dim, sizeof(float) * m * dim, hipMemcpyHostToDevice));
      if (sk_ivf_search(res, ivf_idx, ivf_sp, d_q, m, dim, sc) != 0) { rc = 4; goto done; }
    } else if (bf.search(res, d_base + s * dim, m, dim, sc) != 0) { rc = 4; goto done; }
    hipLaunchKernelGGL(sk_pack_knn, dim3((m + 255) / 256), dim3(256), 0, 0, sc.r_id, m, kk, s, (int64_t)n, d_pack, kin);
    SK_CHECK(hipGetLastError());
    SK_CHECK(hipMemcpy(h_knn.data() + s * kin, d_pack, sizeof(uint32_t) * m * kin, hipMemcpyDeviceToHost));
  }
  *probes = ivf_probes;
  *sampled_recall = est;
done:
#undef SK_CHECK
  bf.close();
  if (ivf_idx) { cuvsIvfFlatIndexDestroy(ivf_idx); }
  if (ivf_ip) { cuvsIvfFlatIndexParamsDestroy(ivf_ip); }
  if (ivf_sp) { cuvsIvfFlatSearchParamsDestroy(ivf_sp); }
  hipFree(d_base); hipFree(d_q); hipFree(d_pack);
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
// The k-NN searches run on a dedicated resources handle that is destroyed before the
// optimize step, so the search workspace it caches never overlaps the graph buffers or
// stays resident for the lifetime of the index.
extern "C" int sk_build_cagra_graph(cuvsResources_t, const float* base, long n, long dim,
                                    int kin, int kout, unsigned int** graph_out)
{
  *graph_out = nullptr;
  if (kin > SK_MAX_K || kout > kin || n <= kin + 1) { return 1; }
  std::vector<uint32_t> h_knn;
  long probes = -1;
  double est = 0;
  cuvsResources_t res;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { return 1; }
  int rc = sk_knn_lists(res, base, n, dim, kin, h_knn, &probes, &est);
  cuvsResourcesDestroy(res);
  if (rc == 0) { rc = sk_optimize_graph(h_knn, n, kin, kout, graph_out); }
  if (rc == 0) {
    fprintf(stderr, "[seekdb_cuvs] cagra graph: n=%ld dim=%ld knn=%s probes=%ld sampled_recall=%.4f\n",
            n, dim, probes > 0 ? "ivf_flat" : "exact", probes > 0 ? probes : 0L, est);
  } else {
    fprintf(stderr, "[seekdb_cuvs] cagra graph build rc=%d: %s\n", rc, cuvsGetLastErrorText());
  }
  return rc;
}

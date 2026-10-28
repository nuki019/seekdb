/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* Search scratch reused across calls (grow-only), guarded by search_mu. */
  pthread_mutex_t        search_mu;
  cuvsCagraSearchParams_t sp;
  size_t                 sp_default_itopk;
  float*                 qd;  size_t qd_cap;
  unsigned*              nd;  size_t nd_cap;
  float*                 dd;  size_t dd_cap;
} sk_cuvs_index;

static void sk_init_search_state(sk_cuvs_index* h)
{
  pthread_mutex_init(&h->search_mu, NULL);
  h->sp = 0; h->sp_default_itopk = 0;
  h->qd = 0; h->qd_cap = 0; h->nd = 0; h->nd_cap = 0; h->dd = 0; h->dd_cap = 0;
}

static void sk_free_search_state(sk_cuvs_index* h)
{
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->qd) cuvsRMMFree(h->res, h->qd, h->qd_cap);
  if (h->nd) cuvsRMMFree(h->res, h->nd, h->nd_cap);
  if (h->dd) cuvsRMMFree(h->res, h->dd, h->dd_cap);
  pthread_mutex_destroy(&h->search_mu);
}

/* Ensures *buf holds at least `bytes` of device memory; keeps the old block when big enough. */
static int sk_reserve(cuvsResources_t res, void** buf, size_t* cap, size_t bytes)
{
  if (*cap >= bytes) return 0;
  if (*buf) { cuvsRMMFree(res, *buf, *cap); *buf = 0; *cap = 0; }
  if (cuvsRMMAlloc(res, buf, bytes) != CUVS_SUCCESS) return 1;
  *cap = bytes;
  return 0;
}

static void sk_host_tensor(DLManagedTensor* t, void* data, long* shape, uint8_t code, uint8_t bits)
{
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCPU; t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = code; t->dl_tensor.dtype.bits = bits; t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape; t->dl_tensor.strides = 0; t->dl_tensor.byte_offset = 0;
  t->manager_ctx = 0; t->deleter = 0;
}

/* seekdb_cuvs_graph.hip: CAGRA graph (n x kout, host) from GPU k-NN search + pruning. */
int sk_build_cagra_graph(cuvsResources_t res, const float* base, long n, long dim,
                         int kin, int kout, unsigned int* out_graph);

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }
  sk_init_search_state(h);

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = 128;
  h->ip->intermediate_graph_degree = 256;
  h->ip->nn_descent_niter = 100;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  {
    /* Build the CAGRA graph with GPU k-NN search + CAGRA pruning (seekdb_cuvs_graph.hip)
     * and attach it with the dataset; fall back to the stock NN-descent build if that fails. */
    long gs[2] = {n, (long)h->ip->graph_degree};
    unsigned int* graph = (unsigned int*)malloc(sizeof(unsigned int) * n * h->ip->graph_degree);
    DLManagedTensor gt;
    int ok = graph != NULL &&
             sk_build_cagra_graph(h->res, base, n, dim, (int)h->ip->intermediate_graph_degree,
                                  (int)h->ip->graph_degree, graph) == 0;
    if (ok) {
      sk_host_tensor(&gt, graph, gs, kDLUInt, 32);
      ok = cuvsCagraIndexFromArgs(h->res, L2Expanded, &gt, &dset, h->idx) == CUVS_SUCCESS &&
           cuvsStreamSync(h->res) == CUVS_SUCCESS;
    }
    free(graph);
    if (!ok) {
      cuvsCagraIndexDestroy(h->idx);
      if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
      if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
    }
  }
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = 0;
  pthread_mutex_lock(&h->search_mu);
  if (!h->sp) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { h->sp = 0; rc=14; goto done; }
    h->sp_default_itopk = h->sp->itopk_size;
  }
  if (sk_reserve(h->res, (void**)&h->qd, &h->qd_cap, sizeof(float)*nq*h->dim))  { rc=10; goto done; }
  if (sk_reserve(h->res, (void**)&h->nd, &h->nd_cap, sizeof(unsigned)*nq*topk)) { rc=11; goto done; }
  if (sk_reserve(h->res, (void**)&h->dd, &h->dd_cap, sizeof(float)*nq*topk))    { rc=12; goto done; }
  if (cudaMemcpy(h->qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    {
      /* Same itopk as before; search_width 8 expands more parents per iteration, which
       * converges in fewer iterations (higher single-query throughput, equal recall). */
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      h->sp->itopk_size = h->sp_default_itopk < required_itopk ? required_itopk : h->sp_default_itopk;
      h->sp->search_width = 8;
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, h->nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, h->dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  pthread_mutex_unlock(&h->search_mu);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  sk_free_search_state(h);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  sk_init_search_state(index);
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

long seekdb_cuvs_graph_degree(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  int64_t degree = 0;
  if (!h || cuvsCagraIndexGetGraphDegree(h->idx, &degree) != CUVS_SUCCESS) return -1;
  return (long)degree;
}

int seekdb_cuvs_get_graph(void* handle, unsigned int* out_graph)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !out_graph) return 1;
  DLManagedTensor g;
  if (cuvsCagraIndexGetGraph(h->idx, &g) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] get graph failed: %s\n", cuvsGetLastErrorText());
    return 2;
  }
  int rc = 0;
  if (g.dl_tensor.ndim != 2 || g.dl_tensor.shape[0] != h->n || g.dl_tensor.dtype.bits != 32) {
    rc = 3;
  } else {
    size_t bytes = sizeof(unsigned) * (size_t)g.dl_tensor.shape[0] * (size_t)g.dl_tensor.shape[1];
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS ||
        cudaMemcpy(out_graph, g.dl_tensor.data, bytes, cudaMemcpyDefault) != cudaSuccess) {
      rc = 4;
    }
  }
  if (g.deleter) g.deleter(&g);
  if (rc) fprintf(stderr, "[seekdb_cuvs] get graph rc=%d\n", rc);
  return rc;
}

void* seekdb_cuvs_from_graph(const float* base, long n, long dim,
                             const unsigned int* graph, long degree)
{
  if (!base || !graph || n <= 0 || dim <= 0 || degree <= 0) return NULL;
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }
  sk_init_search_state(h);
  DLManagedTensor dset, gt;
  long ds[2] = {n, dim}, gs[2] = {n, degree};
  sk_host_tensor(&dset, (void*)base, ds, kDLFloat, 32);
  sk_host_tensor(&gt, (void*)graph, gs, kDLUInt, 32);
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraIndexFromArgs(h->res, L2Expanded, &gt, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  /* Host buffers belong to the caller; make sure the uploads finished before returning. */
  if (cuvsStreamSync(h->res) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  return h;
fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] from_graph failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}

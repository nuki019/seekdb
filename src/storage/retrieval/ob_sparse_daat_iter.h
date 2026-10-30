/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OB_SPARSE_DAAT_ITER_H_
#define OB_SPARSE_DAAT_ITER_H_

#include "ob_i_sparse_retrieval_iter.h"
#include "ob_sparse_utils.h"
#include "storage/access/ob_scan_merge_loser_tree.h"
#include "storage/access/ob_simple_rows_merger.h"

namespace oceanbase
{
namespace storage
{

struct ObSRMergeItem
{
  ObSRMergeItem() : relevance_(0.0), iter_idx_(-1), equal_with_next_(false) {}
  ~ObSRMergeItem() = default;
  TO_STRING_KV(K_(iter_idx), K_(relevance), K_(equal_with_next));

  double relevance_;
  int64_t iter_idx_;
  bool equal_with_next_;
};

struct ObSRMergeCmp
{
  ObSRMergeCmp();
  virtual ~ObSRMergeCmp() {}

  int init(sql::ObDatumMeta id_meta, const ObFixedArray<const ObDatum *, ObIAllocator> *iter_ids);
  int cmp(const ObSRMergeItem &l, const ObSRMergeItem &r, int64_t &cmp_ret);
  bool has_fast_cmp() const { return ID_CMP_GENERIC != id_cmp_mode_; }
  inline int fast_cmp_datum(const ObDatum &l, const ObDatum &r) const { return fast_cmp(id_cmp_mode_, l, r); }
private:
  inline const ObDatum &get_id_datum(const int64_t iter_idx)
  {
    const ObDatum *datum = iter_ids_->at(iter_idx);
    OB_ASSERT(nullptr != datum);
    return *datum;
  }
  enum IdCmpMode
  {
    ID_CMP_GENERIC = 0,
    ID_CMP_BINARY_STR = 1,   // binary collation string (e.g. docid): memcmp + length
    ID_CMP_UINT64 = 2,
    ID_CMP_INT64 = 3,
  };
  static inline int fast_cmp(const IdCmpMode mode, const ObDatum &l, const ObDatum &r)
  {
    int res = 0;
    if (OB_UNLIKELY(l.is_null() || r.is_null())) {
      res = l.is_null() == r.is_null() ? 0 : (l.is_null() ? -1 : 1);  // null first
    } else if (ID_CMP_UINT64 == mode) {
      const uint64_t lv = l.get_uint64();
      const uint64_t rv = r.get_uint64();
      res = lv < rv ? -1 : (lv > rv ? 1 : 0);
    } else if (ID_CMP_INT64 == mode) {
      const int64_t lv = l.get_int();
      const int64_t rv = r.get_int();
      res = lv < rv ? -1 : (lv > rv ? 1 : 0);
    } else if (16 == l.len_ && 16 == r.len_) {
      // docid (tablet id, seq): two big-endian words give the same order as memcmp
      uint64_t lw[2];
      uint64_t rw[2];
      MEMCPY(lw, l.ptr_, sizeof(lw));
      MEMCPY(rw, r.ptr_, sizeof(rw));
      uint64_t lv = __builtin_bswap64(lw[0]);
      uint64_t rv = __builtin_bswap64(rw[0]);
      if (lv == rv) {
        lv = __builtin_bswap64(lw[1]);
        rv = __builtin_bswap64(rw[1]);
      }
      res = lv < rv ? -1 : (lv > rv ? 1 : 0);
    } else {
      const uint32_t ll = l.len_;
      const uint32_t rl = r.len_;
      const int c = MEMCMP(l.ptr_, r.ptr_, ll < rl ? ll : rl);
      res = 0 != c ? (c < 0 ? -1 : 1) : (ll < rl ? -1 : (ll > rl ? 1 : 0));
    }
    return res;
  }
private:
  common::ObDatumCmpFuncType cmp_func_;
  IdCmpMode id_cmp_mode_;
  // TODO: if memory lifetime of docid datum is guaranteed by dim_iters, we can use pointer to datum directly
  //       and avoid deep copy into merge heap here
  const ObFixedArray<const ObDatum *, ObIAllocator> *iter_ids_;
  bool is_inited_;
};

typedef ObSimpleRowsMerger<ObSRMergeItem, ObSRMergeCmp> ObSRSimpleMerger;
typedef ObMergeLoserTree<ObSRMergeItem, ObSRMergeCmp> ObSRLoserTree;
typedef common::ObRowsMerger<ObSRMergeItem, ObSRMergeCmp> ObSRMergeHeap;

// implementation of basic DaaT query processing algorithm primitives
class ObSRDaaTIterImpl : public ObISparseRetrievalMergeIter
{
public:
  ObSRDaaTIterImpl();
  virtual ~ObSRDaaTIterImpl() {}
  virtual int get_next_row() override;
  virtual int get_next_rows(const int64_t capacity, int64_t &count) override;
  int init(
      ObSparseRetrievalMergeParam &iter_param,
      ObIArray<ObISRDaaTDimIter *> &dim_iters,
      ObIAllocator &iter_allocator,
      ObSRDaaTRelevanceCollector &relevance_collector);
  virtual void reuse(const bool switch_tablet = false) override;
  virtual void reset() override;
  virtual int get_query_max_score(double &score) override;
  
  INHERIT_TO_STRING_KV("ObISparseRetrievalMergeIter", ObISparseRetrievalMergeIter,
      K_(next_round_iter_idxes), K_(next_round_cnt));
protected:
  virtual int pre_process();
  virtual int do_one_merge_round(int64_t &count);
  virtual int fill_merge_heap();
  virtual int collect_dims_by_id(const ObDatum *&id_datum, double &relevance, bool &got_valid_id);
  virtual int process_collected_row(const ObDatum &id_datum, const double relevance);
  virtual int filter_on_demand(const int64_t count, const double relevance, bool &need_project);
  virtual int cache_result(int64_t &count, const ObDatum &id_datum, const double relevance);
  virtual int project_results(const int64_t count);
  int init_merge_heap(const int64_t count);
  // Subclasses that only use the default merge primitives may opt in to the heap-free
  // linear merge used for small dimension counts.
  virtual bool support_linear_merge() const { return false; }
  bool can_use_linear_merge() const;
  void reset_linear_merge_state();
  // collect_only: hand the row to process_collected_row() but do not filter / cache it for output
  int linear_merge_round(int64_t &count, const bool collect_only = false);
  static const int64_t LINEAR_MERGE_MAX_DIM_CNT = 8;
protected:
  ObIAllocator *iter_allocator_;
  ObSparseRetrievalMergeParam *iter_param_;
  ObIArray<ObISRDaaTDimIter *> *dim_iters_;
  ObSRMergeCmp merge_cmp_;
  ObSRMergeHeap *merge_heap_;
  ObSRDaaTRelevanceCollector *relevance_collector_;
  ObFixedArray<const ObDatum *, ObIAllocator> iter_domain_ids_; // record every dim iter's output domain id, one (ObDatum *) for one dim iter
  ObFixedArray<ObDocIdExt, ObIAllocator> buffered_domain_ids_; // cache for output
  ObFixedArray<double, ObIAllocator> buffered_relevances_;
  ObFixedArray<int64_t, ObIAllocator> next_round_iter_idxes_;
  int64_t next_round_cnt_;
  void (*set_datum_func_)(ObDatum &, const ObDocIdExt &);
  // "relevance > <const>" match filter: the threshold is evaluated once per scan
  enum GtFilterState { GT_FILTER_UNKNOWN = 0, GT_FILTER_CACHED = 1, GT_FILTER_GENERIC = 2, BOOL_FILTER_CACHED = 3 };
  int8_t gt_filter_state_;
  double gt_filter_threshold_;
  // linear merge state: per dim current (weighted) score and exhaustion flag
  bool lm_exhausted_[LINEAR_MERGE_MAX_DIM_CNT];
  double lm_scores_[LINEAR_MERGE_MAX_DIM_CNT];
private:
  int try_cache_gt_filter_threshold();
  DISALLOW_COPY_AND_ASSIGN(ObSRDaaTIterImpl);
};

} // namespace storage
} // namespace oceanbase

#endif

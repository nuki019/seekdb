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

#define USING_LOG_PREFIX STORAGE

#include "ob_sparse_daat_iter.h"

namespace oceanbase
{
namespace storage
{

ObSRMergeCmp::ObSRMergeCmp()
  : cmp_func_(nullptr),
    id_cmp_mode_(ID_CMP_GENERIC),
    iter_ids_(nullptr),
    is_inited_(false)
{
}

int ObSRMergeCmp::init(
    sql::ObDatumMeta id_meta,
    const ObFixedArray<const ObDatum *, ObIAllocator> *iter_ids)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(iter_ids)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    iter_ids_ = iter_ids;
    common::ObDatumBasicFuncs *basic_funcs = ObDatumFuncs::get_basic_func(id_meta.type_, id_meta.cs_type_);
    cmp_func_ = basic_funcs->null_first_cmp_;
    // Domain ids are compared once per posting entry while merging; avoid the generic
    // collation-aware compare for the common id types whose order is plain byte/number order.
    if (common::ObUInt64Type == id_meta.type_) {
      id_cmp_mode_ = ID_CMP_UINT64;
    } else if (common::ObIntType == id_meta.type_) {
      id_cmp_mode_ = ID_CMP_INT64;
    } else if ((common::ObVarcharType == id_meta.type_ || common::ObCharType == id_meta.type_)
               && common::CS_TYPE_BINARY == id_meta.cs_type_) {
      id_cmp_mode_ = ID_CMP_BINARY_STR;
    } else {
      id_cmp_mode_ = ID_CMP_GENERIC;
    }
    if (OB_ISNULL(cmp_func_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      is_inited_ = true;
    }
  }
  return ret;
}

int ObSRMergeCmp::cmp(
    const ObSRMergeItem &l,
    const ObSRMergeItem &r,
    int64_t &cmp_ret)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (ID_CMP_GENERIC != id_cmp_mode_) {
    cmp_ret = fast_cmp(id_cmp_mode_, *iter_ids_->get_data()[l.iter_idx_], *iter_ids_->get_data()[r.iter_idx_]);
  } else {
    int tmp_ret = 0;
    if (OB_FAIL(cmp_func_(
            get_id_datum(l.iter_idx_), get_id_datum(r.iter_idx_), tmp_ret, nullptr))) {
    } else {
      cmp_ret = tmp_ret;
    }
  }
  return ret;
}

ObSRDaaTIterImpl::ObSRDaaTIterImpl()
  : ObISparseRetrievalMergeIter(),
    iter_allocator_(nullptr),
    iter_param_(nullptr),
    dim_iters_(nullptr),
    merge_cmp_(),
    merge_heap_(nullptr),
    relevance_collector_(nullptr),
    iter_domain_ids_(),
    buffered_domain_ids_(),
    buffered_relevances_(),
    next_round_iter_idxes_(),
    next_round_cnt_(0),
    set_datum_func_(nullptr),
    gt_filter_state_(GT_FILTER_UNKNOWN),
    gt_filter_threshold_(0.0)
{
  reset_linear_merge_state();
}

int ObSRDaaTIterImpl::init(
    ObSparseRetrievalMergeParam &iter_param,
    ObIArray<ObISRDaaTDimIter *> &dim_iters,
    ObIAllocator &iter_allocator,
    ObSRDaaTRelevanceCollector &relevance_collector)
{
  int ret = OB_SUCCESS;
  if (IS_INIT) {
    ret = OB_INIT_TWICE;
  } else if (OB_UNLIKELY(iter_param.max_batch_size_ <= 0)) {
      ret = OB_ERR_UNEXPECTED;
  } else {
    iter_allocator_ = &iter_allocator;
    iter_param_ = &iter_param;
    dim_iters_ = &dim_iters;
    relevance_collector_ = &relevance_collector;
    const int64_t max_batch_size = OB_MAX(iter_param_->eval_ctx_->max_batch_size_, iter_param_->max_batch_size_);
    if (iter_param_->id_proj_expr_->datum_meta_.type_ == common::ObUInt64Type) {
      set_datum_func_ = ObISparseRetrievalMergeIter::set_datum_int;
    } else {
      set_datum_func_ = ObISparseRetrievalMergeIter::set_datum_shallow;
    }
    if (OB_UNLIKELY(dim_iters.count() == 0)) {
    } else if (OB_NOT_NULL(iter_param_->dim_weights_) && dim_iters.count() != iter_param_->dim_weights_->count()) {
      ret = OB_ERR_UNEXPECTED;
    } else if (FALSE_IT(iter_domain_ids_.set_allocator(iter_allocator_))) {
    } else if (OB_FAIL(iter_domain_ids_.init(dim_iters.count()))) {
    } else if (OB_FAIL(iter_domain_ids_.prepare_allocate(dim_iters.count()))) {
    } else if (FALSE_IT(next_round_iter_idxes_.set_allocator(iter_allocator_))) {
    } else if (OB_FAIL(next_round_iter_idxes_.init(dim_iters.count()))) {
    } else if (OB_FAIL(next_round_iter_idxes_.prepare_allocate(dim_iters.count()))) {
    } else if (FALSE_IT(buffered_domain_ids_.set_allocator(iter_allocator_))) {
    } else if (OB_FAIL(buffered_domain_ids_.init(max_batch_size))) {
    } else if (OB_FAIL(buffered_domain_ids_.prepare_allocate(max_batch_size))) {
    } else if (FALSE_IT(buffered_relevances_.set_allocator(iter_allocator_))) {
    } else if (OB_FAIL(buffered_relevances_.init(max_batch_size))) {
    } else if (OB_FAIL(buffered_relevances_.prepare_allocate(max_batch_size))) {
    } else if (OB_FAIL(merge_cmp_.init(iter_param_->id_proj_expr_->datum_meta_, &iter_domain_ids_))) {
    } else if (OB_FAIL(init_merge_heap(dim_iters_->count()))) {
    } else {
      for (int64_t i = 0; i < dim_iters.count(); ++i) {
        next_round_iter_idxes_[i] = i;
      }
    }

    if (OB_SUCC(ret)) {
      next_round_cnt_ = dim_iters.count();
      input_row_cnt_ = 0;
      output_row_cnt_ = 0;
      is_inited_ = true;
    }
  }
  return ret;
}

void ObSRDaaTIterImpl::reset()
{
  if (OB_NOT_NULL(merge_heap_)) {
    merge_heap_->~ObSRMergeHeap();
    merge_heap_ = nullptr;
  }
  if (OB_NOT_NULL(relevance_collector_)) {
    relevance_collector_->reset();
    relevance_collector_ = nullptr;
  }
  iter_domain_ids_.reset();
  buffered_domain_ids_.reset();
  buffered_relevances_.reset();
  next_round_iter_idxes_.reset();
  next_round_cnt_ = 0;
  input_row_cnt_ = 0;
  output_row_cnt_ = 0;
  is_inited_ = false;
}

void ObSRDaaTIterImpl::reuse(const bool switch_tablet)
{
  if (OB_NOT_NULL(dim_iters_)) {
    if (OB_NOT_NULL(merge_heap_)) {
      merge_heap_->reuse();
      merge_heap_->open(dim_iters_->count());
    }
    next_round_cnt_ = dim_iters_->count();
    for (int64_t i = 0; i < next_round_cnt_; ++i) {
      next_round_iter_idxes_[i] = i;
    }
  }
  if (OB_NOT_NULL(relevance_collector_)) {
    relevance_collector_->reuse();
  }
  input_row_cnt_ = 0;
  output_row_cnt_ = 0;
  gt_filter_state_ = GT_FILTER_UNKNOWN; // exec params may change on rescan
  reset_linear_merge_state();
}

  int ObSRDaaTIterImpl::get_query_max_score(double &score)
  {
    int ret = OB_SUCCESS;
    score = 0.0;
    for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
      ObISRDaaTDimIter *dim_iter = dim_iters_->at(i);
      double dim_score = 0.0;
      if (OB_ISNULL(dim_iter) || OB_ISNULL(iter_param_->dim_weights_)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(dim_iter->get_dim_max_score(dim_score))) {
      } else {
        score += dim_score * iter_param_->field_boost_ * iter_param_->dim_weights_->at(i);
      }
    }

    return ret;
  }

int ObSRDaaTIterImpl::get_next_row()
{
  int ret = OB_SUCCESS;
  int64_t count = 0;
  if (OB_FAIL(get_next_rows(1, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    } else if (OB_UNLIKELY(count != 0)) {
      ret = OB_ERR_UNEXPECTED;
    }
  } else if (OB_UNLIKELY(count != 1)) {
    ret = OB_ERR_UNEXPECTED;
  }
  return ret;
}

int ObSRDaaTIterImpl::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(0 == capacity)) {
    count = 0;
  } else if (0 == dim_iters_->count()) {
    ret = OB_ITER_END;
  } else if (iter_param_->limit_param_->is_valid() && output_row_cnt_ >= iter_param_->limit_param_->limit_) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(pre_process())) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    } else {
      count = 0;
    }
  } else {
    count = 0;
    const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
    const bool use_linear_merge = can_use_linear_merge();
    while (OB_SUCC(ret) && count < real_capacity) {
      if (OB_FAIL(use_linear_merge ? linear_merge_round(count) : do_one_merge_round(count))) {
        if (OB_UNLIKELY(OB_ITER_END != ret)) {
        }
      }
    }

    if (OB_SUCC(ret) || OB_ITER_END == ret) {
      if (count > 0 && OB_FAIL(project_results(count))) {
      }
    }

  }
  return ret;
}

int ObSRDaaTIterImpl::pre_process()
{
  return OB_NOT_IMPLEMENT;
}

bool ObSRDaaTIterImpl::can_use_linear_merge() const
{
  return support_linear_merge()
      && merge_cmp_.has_fast_cmp()
      && OB_NOT_NULL(dim_iters_)
      && dim_iters_->count() > 0
      && dim_iters_->count() <= LINEAR_MERGE_MAX_DIM_CNT;
}

void ObSRDaaTIterImpl::reset_linear_merge_state()
{
  for (int64_t i = 0; i < LINEAR_MERGE_MAX_DIM_CNT; ++i) {
    lm_exhausted_[i] = false;
    lm_scores_[i] = 0.0;
  }
}

// Heap-free equivalent of fill_merge_heap() + collect_dims_by_id() for a few dimensions:
// advance the dims consumed by the previous round, pick the smallest current id by a linear
// scan, and collect every dim positioned on it (in ascending dim order).
int ObSRDaaTIterImpl::linear_merge_round(int64_t &count)
{
  int ret = OB_SUCCESS;
  const int64_t dim_cnt = dim_iters_->count();
  ObISRDaaTDimIter *const *dim_iters = dim_iters_->get_data();
  int64_t *round_idxes = next_round_iter_idxes_.get_data();
  const ObDatum **domain_ids = iter_domain_ids_.get_data();
  const double *dim_weights = OB_NOT_NULL(iter_param_->dim_weights_) ? iter_param_->dim_weights_->get_data() : nullptr;
  for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
    const int64_t iter_idx = round_idxes[i];
    ObISRDaaTDimIter *dim_iter = dim_iters[iter_idx];
    double score = 0.0;
    if (OB_ISNULL(dim_iter)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dim_iter->get_next_row())) {
      if (OB_UNLIKELY(OB_ITER_END != ret)) {
        LOG_WARN("fail to try load next batch dimension data", K(ret));
      } else {
        lm_exhausted_[iter_idx] = true;
        ret = OB_SUCCESS;
      }
    } else if (OB_FAIL(dim_iter->get_curr_score(score))) {
    } else if (OB_FAIL(dim_iter->get_curr_id(domain_ids[iter_idx]))) {
    } else {
      lm_scores_[iter_idx] = nullptr != dim_weights
          ? score * iter_param_->field_boost_ * dim_weights[iter_idx] : score;
    }
  }
  if (OB_FAIL(ret)) {
  } else {
    next_round_cnt_ = 0;
    int64_t min_idx = -1;
    for (int64_t i = 0; i < dim_cnt; ++i) {
      if (!lm_exhausted_[i]
          && (min_idx < 0 || merge_cmp_.fast_cmp_datum(*domain_ids[i], *domain_ids[min_idx]) < 0)) {
        min_idx = i;
      }
    }
    if (min_idx < 0) {
      ret = OB_ITER_END;
    } else {
      // dims before min_idx are strictly greater, so collection starts at min_idx
      for (int64_t i = min_idx; OB_SUCC(ret) && i < dim_cnt; ++i) {
        if (!lm_exhausted_[i]
            && (i == min_idx || 0 == merge_cmp_.fast_cmp_datum(*domain_ids[i], *domain_ids[min_idx]))) {
          if (OB_FAIL(relevance_collector_->collect_one_dim(i, lm_scores_[i]))) {
          } else {
            round_idxes[next_round_cnt_++] = i;
          }
        }
      }
      const ObDatum *id_datum = domain_ids[min_idx];
      double relevance = 0.0;
      bool need_project = false;
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(relevance_collector_->get_result(relevance, need_project))) {
      } else if (need_project && OB_FAIL(process_collected_row(*id_datum, relevance))) {
      } else if (need_project && OB_FAIL(filter_on_demand(count, relevance, need_project))) {
      } else if (need_project && OB_FAIL(cache_result(count, *id_datum, relevance))) {
        if (OB_ITER_END != ret) {
        }
      }
    }
  }
  return ret;
}

int ObSRDaaTIterImpl::do_one_merge_round(int64_t &count)
{
  int ret = OB_SUCCESS;
  bool need_project = true;
  double relevance = 0.0;
  const ObDatum *id_datum = nullptr;
  if (OB_FAIL(fill_merge_heap())) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  } else if (OB_FAIL(collect_dims_by_id(id_datum, relevance, need_project))) {
  } else if (need_project && OB_FAIL(filter_on_demand(count, relevance, need_project))) {
  } else if (need_project && OB_FAIL(cache_result(count, *id_datum, relevance))) {
    if (OB_ITER_END != ret) {
    }
  }
  return ret;
}

int ObSRDaaTIterImpl::fill_merge_heap()
{
  int ret = OB_SUCCESS;
  ObSRMergeItem item;
  // hot loop: index the fixed-size arrays directly (indexes come from next_round_iter_idxes_,
  // which only ever holds valid dim iter positions)
  ObISRDaaTDimIter *const *dim_iters = dim_iters_->get_data();
  const int64_t *round_idxes = next_round_iter_idxes_.get_data();
  const ObDatum **domain_ids = iter_domain_ids_.get_data();
  const double *dim_weights = OB_NOT_NULL(iter_param_->dim_weights_) ? iter_param_->dim_weights_->get_data() : nullptr;
  for (int64_t i = 0; OB_SUCC(ret) && i < next_round_cnt_; ++i) {
    const int64_t iter_idx = round_idxes[i];
    ObISRDaaTDimIter *dim_iter = nullptr;
    if (OB_ISNULL(dim_iter = dim_iters[iter_idx])) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(dim_iter->get_next_row())) {
      if (OB_UNLIKELY(OB_ITER_END != ret)) {
        LOG_WARN("fail to try load next batch dimension data", K(ret));\
      } else {
        ret = OB_SUCCESS;
      }
    } else if (OB_FAIL(dim_iter->get_curr_score(item.relevance_))) {
    } else if (nullptr != dim_weights && FALSE_IT(item.relevance_ = item.relevance_ * iter_param_->field_boost_ * dim_weights[iter_idx])) {
    } else if (OB_FAIL(dim_iter->get_curr_id(domain_ids[iter_idx]))) {
    } else if (FALSE_IT(item.iter_idx_ = iter_idx)) {
    } else if (OB_FAIL(merge_heap_->push(item))) {
    } else {
      LOG_DEBUG("push item to merge heap", K(ret), K(i), K(iter_idx), K(next_round_cnt_), K(item), K(iter_domain_ids_[iter_idx]));
    }
  }

  if (OB_FAIL(ret)) {
  } else if (merge_heap_->empty()) {
    ret = OB_ITER_END;
  } else if (0 != next_round_cnt_ && OB_FAIL(merge_heap_->rebuild())) {
  } else {
    next_round_cnt_ = 0;
  }
  return ret;
}

int ObSRDaaTIterImpl::collect_dims_by_id(const ObDatum *&id_datum, double &relevance, bool &got_valid_id)
{
  int ret = OB_SUCCESS;

  const ObSRMergeItem *top_item = nullptr;
  bool curr_doc_end = false;
  int64_t iter_idx = 0;
  relevance = 0.0;
  got_valid_id = false;

  while (OB_SUCC(ret) && !merge_heap_->empty() && !curr_doc_end) {
    if (merge_heap_->is_unique_champion()) {
      curr_doc_end = true;
    }
    if (OB_FAIL(merge_heap_->top(top_item))) {
    } else if (OB_FAIL(relevance_collector_->collect_one_dim(top_item->iter_idx_, top_item->relevance_))) {
    } else if (FALSE_IT(iter_idx = top_item->iter_idx_)) {
    } else if (OB_FAIL(merge_heap_->pop())) {
    } else {
      next_round_iter_idxes_.get_data()[next_round_cnt_++] = iter_idx;
    }
  }

  if (OB_SUCC(ret)) {
    id_datum = iter_domain_ids_.get_data()[iter_idx];
    if (OB_ISNULL(id_datum)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(relevance_collector_->get_result(relevance, got_valid_id))) {
    } else if (got_valid_id && OB_FAIL(process_collected_row(*id_datum, relevance))) {
    }
  }

  return ret;
}

int ObSRDaaTIterImpl::process_collected_row(const ObDatum &id_datum, const double relevance)
{
  return OB_SUCCESS;
}

int ObSRDaaTIterImpl::filter_on_demand(const int64_t count, const double relevance, bool &need_project)
{
  int ret = OB_SUCCESS;
  if (!iter_param_->need_filter()) {
    need_project = true;
    return OB_SUCCESS;
  }
  if (OB_UNLIKELY(GT_FILTER_UNKNOWN == gt_filter_state_) && OB_FAIL(try_cache_gt_filter_threshold())) {
    return ret;
  } else if (GT_FILTER_CACHED == gt_filter_state_) {
    // relevance itself is projected later in project_results()
    need_project = relevance > gt_filter_threshold_;
    return ret;
  } else if (BOOL_FILTER_CACHED == gt_filter_state_) {
    need_project = (0 != relevance);
    return ret;
  }
  ObExpr *filter_expr = iter_param_->filter_expr_;
  ObExpr *relevance_proj_expr = iter_param_->relevance_proj_expr_;
  ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;
  ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx);

  if (eval_ctx->is_vectorized()) {
    if (OB_UNLIKELY(count >= eval_ctx->max_batch_size_)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      guard.set_batch_idx(count);
      ObDatum &relevance_proj_datum = relevance_proj_expr->locate_datum_for_write(*eval_ctx);
      relevance_proj_datum.set_double(relevance);
      relevance_proj_expr->get_evaluated_flags(*eval_ctx).set(count);
      relevance_proj_expr->set_evaluated_projected(*eval_ctx);
      if (count > 0 &&
          T_OP_GT == filter_expr->type_ &&
          2 == filter_expr->arg_cnt_ &&
          filter_expr->args_[0] == relevance_proj_expr &&
          common::ObDoubleType == filter_expr->args_[1]->datum_meta_.type_ &&
          filter_expr->args_[1]->get_evaluated_flags(*eval_ctx).at(0)) {
        const ObDatum &rhs = filter_expr->args_[1]->locate_expr_datum(*eval_ctx, 0);
        if (!rhs.is_null()) {
          need_project = (relevance > rhs.get_double());
          return OB_SUCCESS;
        }
      }
      ObDatum *filter_res = nullptr;
      filter_expr->clear_evaluated_flag(*eval_ctx);
      if (OB_FAIL(filter_expr->eval(*eval_ctx, filter_res))) {
      } else {
        need_project = !(filter_res->is_null() || 0 == filter_res->get_int());
      }
    }
  } else if (OB_UNLIKELY(0 != count)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    guard.set_batch_idx(0);
    ObDatum &relevance_proj_datum = relevance_proj_expr->locate_datum_for_write(*eval_ctx);
    relevance_proj_datum.set_double(relevance);
    relevance_proj_expr->set_evaluated_projected(*eval_ctx);
  }
  return ret;
}

int ObSRDaaTIterImpl::try_cache_gt_filter_threshold()
{
  int ret = OB_SUCCESS;
  gt_filter_state_ = GT_FILTER_GENERIC;
  ObExpr *filter_expr = iter_param_->filter_expr_;
  ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;
  if (OB_NOT_NULL(filter_expr) && OB_NOT_NULL(eval_ctx)
      && eval_ctx->is_vectorized()
      && T_OP_BOOL == filter_expr->type_
      && 1 == filter_expr->arg_cnt_
      && filter_expr->args_[0] == iter_param_->relevance_proj_expr_
      && common::ObDoubleType == filter_expr->args_[0]->datum_meta_.type_) {
    // natural language match filter: bool(relevance), i.e. relevance != 0 (relevance is never null)
    gt_filter_state_ = BOOL_FILTER_CACHED;
  } else if (OB_NOT_NULL(filter_expr) && OB_NOT_NULL(eval_ctx)
      && eval_ctx->is_vectorized()
      && T_OP_GT == filter_expr->type_
      && 2 == filter_expr->arg_cnt_
      && filter_expr->args_[0] == iter_param_->relevance_proj_expr_
      && common::ObDoubleType == filter_expr->args_[1]->datum_meta_.type_
      && filter_expr->args_[1]->is_const_expr()) {
    ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx);
    guard.set_batch_idx(0);
    ObDatum *rhs = nullptr;
    if (OB_FAIL(filter_expr->args_[1]->eval(*eval_ctx, rhs))) {
      LOG_WARN("failed to eval match filter threshold", K(ret));
    } else if (OB_NOT_NULL(rhs) && !rhs->is_null()) {
      gt_filter_threshold_ = rhs->get_double();
      gt_filter_state_ = GT_FILTER_CACHED;
    }
  }
  return ret;
}

int ObSRDaaTIterImpl::cache_result(int64_t &count, const ObDatum &id_datum, const double relevance)
{
  int ret = OB_SUCCESS;
  const common::ObLimitParam *limit_param= iter_param_->limit_param_;
  const int64_t limit = limit_param->limit_;
  const int64_t offset = limit_param->offset_;
  ++input_row_cnt_;
  if (limit_param->is_valid() && input_row_cnt_ <= offset) {
    // TODO: Maybe we should not process offset logic here
    // don't need to project
  } else if (OB_UNLIKELY(count >= buffered_domain_ids_.count() || count >= buffered_relevances_.count())) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    buffered_domain_ids_[count].from_datum(id_datum);
    buffered_relevances_[count] = relevance;
    ++count;
    ++output_row_cnt_;
    if (limit_param->is_valid() && output_row_cnt_ >= limit) {
      ret = OB_ITER_END;
    }
  }
  return ret;
}

int ObSRDaaTIterImpl::project_results(const int64_t count)
{
  int ret = OB_SUCCESS;
  ObExpr *relevance_proj_expr = iter_param_->relevance_proj_expr_;
  ObExpr *id_proj_expr = iter_param_->id_proj_expr_;
  ObEvalCtx *eval_ctx = iter_param_->eval_ctx_;

  if (eval_ctx->is_vectorized()) {
    ObDatum *id_datums = id_proj_expr->locate_datums_for_update(*eval_ctx, count);
    for (int64_t i = 0; i < count; ++i) {
      set_datum_func_(id_datums[i], buffered_domain_ids_[i]);
    }
    id_proj_expr->get_evaluated_flags(*eval_ctx).set_all(count);
    id_proj_expr->set_evaluated_projected(*eval_ctx);
    if (iter_param_->need_project_relevance()) {
      ObDatum *rel_datums = relevance_proj_expr->locate_datums_for_update(*eval_ctx, count);
      for (int64_t i = 0; i < count; ++i) {
        rel_datums[i].set_double(buffered_relevances_[i]);
      }
      relevance_proj_expr->get_evaluated_flags(*eval_ctx).set_all(count);
      relevance_proj_expr->set_evaluated_projected(*eval_ctx);
    }
  } else if (OB_UNLIKELY(1 != count)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    ObEvalCtx::BatchInfoScopeGuard guard(*eval_ctx);
    guard.set_batch_idx(0);
    ObDatum &id_proj_datum = id_proj_expr->locate_datum_for_write(*eval_ctx);
    set_datum_func_(id_proj_datum, buffered_domain_ids_[0]);
    id_proj_expr->set_evaluated_projected(*eval_ctx);
    ObDatum &relevance_proj_datum = relevance_proj_expr->locate_datum_for_write(*eval_ctx);
    relevance_proj_datum.set_double(buffered_relevances_[0]);
    relevance_proj_expr->set_evaluated_projected(*eval_ctx);
  }
  return ret;
}

int ObSRDaaTIterImpl::init_merge_heap(const int64_t count)
{
  int ret = OB_SUCCESS;
  ObSRSimpleMerger *simple_merge = nullptr;
  ObSRMergeHeap *loser_tree = nullptr;
  if (count <= ObSRSimpleMerger::USE_SIMPLE_MERGER_MAX_TABLE_CNT) {
    if (OB_ISNULL(simple_merge = OB_NEWx(ObSRSimpleMerger, iter_allocator_, merge_cmp_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      merge_heap_ = simple_merge;
    }
  } else if (OB_ISNULL(loser_tree = OB_NEWx(ObSRLoserTree, iter_allocator_, merge_cmp_))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    merge_heap_ = loser_tree;
  }
  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(merge_heap_->init(count, count, *iter_allocator_))) {
  } else if (OB_FAIL(merge_heap_->open(count))) {
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase

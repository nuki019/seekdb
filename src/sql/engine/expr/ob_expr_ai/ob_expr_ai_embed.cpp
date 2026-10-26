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

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"
#include "sql/engine/ob_operator.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{
ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (ob_is_text_tc(types_stack[CONTENT_IDX].get_type())) {
      types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    } else {
      types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
      types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    }
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  if (expr.is_batch_result() && ctx.get_batch_size() > 1) {
    const ObBitVector *skip_vec = nullptr;
    ObOpKitStore &kit_store = ctx.exec_ctx_.get_kit_store();
    for (int64_t i = 0; i < kit_store.size_; ++i) {
      if (OB_NOT_NULL(kit_store.kits_) && OB_NOT_NULL(kit_store.kits_[i].op_)) {
        ObOperator *op = kit_store.kits_[i].op_;
        if (&op->get_eval_ctx() == &ctx
            && OB_NOT_NULL(op->get_brs().skip_)
            && op->get_brs().size_ == ctx.get_batch_size()) {
          skip_vec = op->get_brs().skip_;
          break;
        }
      }
    }
    if (OB_NOT_NULL(skip_vec)) {
      ret = eval_ai_embed_batch(expr, ctx, *skip_vec, ctx.get_batch_size());
    } else {
      ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
      ObIAllocator &tmp_alloc = tmp_alloc_g.get_allocator();
      void *skip_mem = tmp_alloc.alloc(ObBitVector::memory_size(ctx.get_batch_size()));
      if (OB_ISNULL(skip_mem)) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        ObBitVector *tmp_skip = to_bit_vector(skip_mem);
        tmp_skip->init(ctx.get_batch_size());
        for (int64_t k = 0; k < ctx.get_batch_idx(); ++k) {
          tmp_skip->set(k);
        }
        ret = eval_ai_embed_batch(expr, ctx, *tmp_skip, ctx.get_batch_size());
      }
    }
    return ret;
  }

  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (OB_ISNULL(arg_model_id) || arg_model_id->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or content is null");
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id;
    ObString content;
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                  ctx.exec_ctx_, temp_allocator, *arg_model_id,
                  expr.args_[MODEL_IDX]->datum_meta_,
                  expr.args_[MODEL_IDX]->obj_meta_.has_lob_header(), model_id))) {
    } else if (model_id.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    }
    int64_t dim = 0;
    ObJsonInt *dim_json = nullptr;
    ObJsonObject *config = nullptr;
    if (OB_FAIL(ret)) {
    } else if (expr.arg_cnt_ == 3 && OB_NOT_NULL(arg_dim)) {
      if (arg_dim->is_null() || (dim = arg_dim->get_int()) <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        res.set_null();
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, dim, dim_json))) {
      } else if (OB_FAIL(config->add("dimensions", dim_json))) {
      }
    }
    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(arg_content) || arg_content->is_null()) {
      res.set_null();
    } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                         ctx.exec_ctx_, temp_allocator, *arg_content,
                         expr.args_[CONTENT_IDX]->datum_meta_,
                         expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), content))) {
    } else if (content.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      res.set_null();
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } else {
      ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
      ObString result;
      if (OB_FAIL(model.call_dense_embedding(content, config, result))) {
      } else if (OB_FAIL(ObAIFuncUtils::set_string_result(expr, ctx, res, result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr,
                                       ObEvalCtx &ctx,
                                       const ObBitVector &skip,
                                       const int64_t batch_size)
{
  INIT_SUCC(ret);
  ObDatumVector res_datum = expr.locate_expr_datumvector(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
  MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
  lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
  ObEvalCtx::BatchInfoScopeGuard batch_info_guard(ctx);
  batch_info_guard.set_batch_size(batch_size);
  bool got_null = false;

  if (OB_FAIL(expr.args_[MODEL_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else if (OB_FAIL(expr.args_[CONTENT_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else if (expr.arg_cnt_ == 3 && OB_FAIL(expr.args_[DIM_IDX]->eval_batch(ctx, skip, batch_size))) {
  } else {
    ObDatumVector model_id_datums = expr.args_[MODEL_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector content_datums = expr.args_[CONTENT_IDX]->locate_expr_datumvector(ctx);
    ObDatumVector dim_datums = (expr.arg_cnt_ == 3)
        ? expr.args_[DIM_IDX]->locate_expr_datumvector(ctx)
        : ObDatumVector();

    ObString batch_model_id;
    int64_t batch_dim = 0;
    bool has_batch_meta = false;
    ObArray<ObString> batch_contents;
    ObArray<int64_t> batch_row_idxs;

    auto flush_sub_batch = [&]() -> int {
      if (!batch_contents.empty()) {
        ObAIFuncExprInfo *info = nullptr;
        share::ObAiModelEndpointInfo resolved_endpoint;
        const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
        query::ObIAiEndpointResolver *endpoint_resolver =
            ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
        ObJsonObject *config = nullptr;
        ObJsonInt *dim_json = nullptr;
        if (batch_dim > 0) {
          if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(temp_allocator, config))) {
          } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(temp_allocator, batch_dim, dim_json))) {
          } else if (OB_FAIL(config->add("dimensions", dim_json))) {
          }
        }
        if (OB_SUCC(ret)) {
          ObArray<ObString> batch_results;
          if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, batch_model_id, info))) {
          } else if (OB_ISNULL(endpoint_resolver)) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                         batch_model_id, temp_allocator, resolved_endpoint))) {
          } else {
            ObAIFuncModel model(temp_allocator, *info, *endpoint_info);
            if (OB_FAIL(model.call_dense_embedding_vector_v2(batch_contents, config, batch_results))) {
            } else if (batch_results.count() != batch_row_idxs.count()) {
              ret = OB_INVALID_DATA;
            } else {
              for (int64_t k = 0; OB_SUCC(ret) && k < batch_row_idxs.count(); ++k) {
                int64_t row_idx = batch_row_idxs.at(k);
                batch_info_guard.set_batch_idx(row_idx);
                ObString &res_str = batch_results.at(k);
                ObTextStringDatumResult text_result(expr.datum_meta_.type_, &expr, &ctx, res_datum.at(row_idx));
                if (OB_FAIL(text_result.init_with_batch_idx(res_str.length(), row_idx))) {
                } else if (OB_FAIL(text_result.append(res_str))) {
                } else {
                  text_result.set_result();
                  eval_flags.set(row_idx);
                }
              }
            }
          }
        }
        batch_contents.reuse();
        batch_row_idxs.reuse();
      }
      return ret;
    };

    for (int64_t j = 0; OB_SUCC(ret) && j < batch_size; ++j) {
      if (skip.at(j) || eval_flags.at(j)) {
        continue;
      }
      batch_info_guard.set_batch_idx(j);
      ObDatum *model_datum = model_id_datums.at(j);
      ObDatum *content_datum = content_datums.at(j);
      ObDatum *dim_datum = (expr.arg_cnt_ == 3) ? dim_datums.at(j) : nullptr;
      ObString row_model_id;
      int64_t row_dim = 0;

      if (OB_ISNULL(model_datum) || model_datum->is_null()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or content is null");
      } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                           ctx.exec_ctx_, temp_allocator, *model_datum,
                           expr.args_[MODEL_IDX]->datum_meta_,
                           expr.args_[MODEL_IDX]->obj_meta_.has_lob_header(), row_model_id))) {
      } else if (row_model_id.empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
      } else if (expr.arg_cnt_ == 3) {
        if (OB_ISNULL(dim_datum) || dim_datum->is_null() || (row_dim = dim_datum->get_int()) <= 0) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
        }
      }

      if (OB_FAIL(ret)) {
      } else if (OB_ISNULL(content_datum) || content_datum->is_null()) {
        res_datum.at(j)->set_null();
        eval_flags.set(j);
        got_null = true;
      } else {
        ObString row_content;
        if (OB_FAIL(ObTextStringHelper::read_real_string_data(
                      ctx.exec_ctx_, temp_allocator, *content_datum,
                      expr.args_[CONTENT_IDX]->datum_meta_,
                      expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), row_content))) {
        } else if (row_content.empty()) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, model id or input is empty");
        } else {
          if (has_batch_meta && (row_model_id != batch_model_id || row_dim != batch_dim)) {
            if (OB_FAIL(flush_sub_batch())) {
            }
          }
          if (OB_SUCC(ret)) {
            batch_model_id = row_model_id;
            batch_dim = row_dim;
            has_batch_meta = true;
            if (OB_FAIL(batch_contents.push_back(row_content))
                || OB_FAIL(batch_row_idxs.push_back(j))) {
            }
          }
        }
      }
    }

    if (OB_SUCC(ret) && !batch_contents.empty()) {
      if (OB_FAIL(flush_sub_batch())) {
      }
    }
  }

  if (OB_FAIL(ret)) {
    for (int64_t j = 0; j < batch_size; ++j) {
      if (!skip.at(j)) {
        res_datum.at(j)->set_null();
        eval_flags.unset(j);
      }
    }
    expr.get_eval_info(ctx).notnull_ = false;
  } else if (got_null) {
    expr.get_eval_info(ctx).notnull_ = false;
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase

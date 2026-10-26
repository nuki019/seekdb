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
#include "ob_expr_ai_rerank.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "lib/utility/utility.h"
#include "common/json_type/ob_json_common.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"
#include <cmath>
 
using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

namespace
{
bool extract_json_int(const ObJsonNode *node, int64_t &val)
{
  bool ok = false;
  val = -1;
  if (OB_NOT_NULL(node)) {
    ObJsonNodeType type = node->json_type();
    if (type == ObJsonNodeType::J_INT ||
        type == ObJsonNodeType::J_OINT ||
        type == ObJsonNodeType::J_OLONG) {
      val = node->get_int();
      ok = true;
    } else if (type == ObJsonNodeType::J_UINT) {
      uint64_t uval = node->get_uint();
      if (uval <= static_cast<uint64_t>(INT64_MAX)) {
        val = static_cast<int64_t>(uval);
        ok = true;
      }
    }
  }
  return ok;
}

bool extract_json_score(const ObJsonNode *node, double &score)
{
  bool ok = false;
  score = 0.0;
  if (OB_NOT_NULL(node) && share::ObJsonHelper::is_number_type(node)) {
    ObJsonNodeType type = node->json_type();
    if (type == ObJsonNodeType::J_DOUBLE || type == ObJsonNodeType::J_ODOUBLE) {
      score = node->get_double();
      ok = std::isfinite(score);
    } else if (type == ObJsonNodeType::J_OFLOAT) {
      score = static_cast<double>(node->get_float());
      ok = std::isfinite(score);
    } else if (type == ObJsonNodeType::J_INT || type == ObJsonNodeType::J_OINT || type == ObJsonNodeType::J_OLONG) {
      score = static_cast<double>(node->get_int());
      ok = true;
    } else if (type == ObJsonNodeType::J_UINT) {
      score = static_cast<double>(node->get_uint());
      ok = true;
    } else if (type == ObJsonNodeType::J_DECIMAL || type == ObJsonNodeType::J_ODECIMAL) {
      number::ObNumber decimal_val = node->get_decimal_data();
      char decimal_buf[512];
      int64_t pos = 0;
      if (OB_SUCCESS == decimal_val.format(decimal_buf, sizeof(decimal_buf), pos, -1)) {
        decimal_buf[pos] = '\0';
        score = atof(decimal_buf);
        ok = std::isfinite(score);
      }
    }
  }
  return ok;
}

struct RerankSortableItem
{
  RerankSortableItem()
    : obj_(nullptr), score_(0.0), index_(-1), has_index_(false), order_pos_(0)
  {}
  RerankSortableItem(ObJsonObject *obj, double score, int64_t index, bool has_index, int64_t order_pos)
    : obj_(obj), score_(score), index_(index), has_index_(has_index), order_pos_(order_pos)
  {}
  ObJsonObject *obj_;
  double score_;
  int64_t index_;
  bool has_index_;
  int64_t order_pos_;
  TO_STRING_KV(KP_(obj), K_(score), K_(index), K_(has_index), K_(order_pos));
};

struct RerankItemComparator
{
  bool operator()(const RerankSortableItem &a, const RerankSortableItem &b) const
  {
    if (a.score_ != b.score_) {
      return a.score_ > b.score_;
    }
    if (a.has_index_ && b.has_index_ && a.index_ != b.index_) {
      return a.index_ < b.index_;
    }
    return a.order_pos_ < b.order_pos_;
  }
};
} // anonymous namespace

ObExprAIRerank::ObExprAIRerank(common::ObIAllocator &alloc)
: ObFuncExprOperator(alloc, 
                    T_FUN_SYS_AI_RERANK, 
                    N_AI_RERANK, 
                    MORE_THAN_ZERO,
                    NOT_VALID_FOR_GENERATED_COL, 
                    NOT_ROW_DIMENSION) 
{
}

ObExprAIRerank::~ObExprAIRerank() 
{
}

int ObExprAIRerank::calc_result_typeN(ObExprResType &type,
                                    ObExprResType *types_stack,
                                    int64_t param_num,
                                    common::ObExprTypeCtx &type_ctx) const {
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num < 3 || param_num > 4)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    if (!ob_is_string_type(types_stack[MODEL_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, model key must be string type");
    } else if (!ob_is_string_type(types_stack[QUERY_IDX].get_type())) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, query must be string type");
    } else {
      if (ob_is_string_tc(types_stack[MODEL_IDX].get_type())) {
        types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
      }
      types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      if (ob_is_string_tc(types_stack[QUERY_IDX].get_type())) {
        types_stack[QUERY_IDX].set_calc_type(ObVarcharType);
      }
      types_stack[QUERY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    }
    ObObjType in_type = types_stack[DOCUMENTS_IDX].get_type();
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObJsonExprHelper::is_valid_for_json(types_stack, DOCUMENTS_IDX, N_AI_RERANK))) {
    } else if (ob_is_string_type(in_type) && types_stack[DOCUMENTS_IDX].get_collation_type() != CS_TYPE_BINARY) {
      if (types_stack[DOCUMENTS_IDX].get_charset_type() != CHARSET_UTF8MB4) {
        types_stack[DOCUMENTS_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      }
    }
    if (OB_FAIL(ret)) {
    } else if (param_num == 4) {
      ObObjType doc_key_type = types_stack[DOC_KEY_IDX].get_type();
      if (doc_key_type == ObNullType) {
      } else if (doc_key_type == ObJsonType) {
      } else if (ob_is_string_type(doc_key_type)) {
        if (ob_is_string_tc(doc_key_type)) {
          types_stack[DOC_KEY_IDX].set_calc_type(ObVarcharType);
        }
        types_stack[DOC_KEY_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
      } else {  
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key must be string type");
      }
    }

    if (OB_SUCC(ret)) {
      type.set_json();
      type.set_length((ObAccuracy::DDL_DEFAULT_ACCURACY[ObJsonType]).get_length());
    }
  }
  return ret;
}

int ObExprAIRerank::eval_ai_rerank(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_query = nullptr;
  ObDatum *arg_documents = nullptr;
  ObDatum *arg_doc_key = nullptr;
  if (OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_query, arg_documents, arg_doc_key))) {
  } else if (arg_model_id->is_null() || arg_query->is_null() || arg_documents->is_null()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, model id or query or documents is null");
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_RERANK));
    ObAIFuncExprInfo *info = nullptr;
    share::ObAiModelEndpointInfo resolved_endpoint;
    const share::ObAiModelEndpointInfo *endpoint_info = &resolved_endpoint;
    query::ObIAiEndpointResolver *endpoint_resolver =
        ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
    ObString model_id;
    ObString query;
    ObString doc_key;
    bool has_doc_key = false;
    ObArray<ObString> header_array;
    ObJsonArray *document_array = nullptr;
    ObJsonArray *result_array = nullptr;
    ObIJsonBase *j_base = nullptr;
    int64_t batch_size = 20; // max batch size for rerank
    bool is_null_result = false;
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, temp_allocator, *arg_model_id,
                  expr.args_[MODEL_IDX]->datum_meta_, expr.args_[MODEL_IDX]->obj_meta_.has_lob_header(), model_id))) {
    } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, temp_allocator, *arg_query,
                         expr.args_[QUERY_IDX]->datum_meta_, expr.args_[QUERY_IDX]->obj_meta_.has_lob_header(), query))) {
    } else if (model_id.empty() || query.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, model id or query is empty");
      res.set_null();
    } else if (expr.arg_cnt_ == 4 && OB_NOT_NULL(arg_doc_key) && !arg_doc_key->is_null()) {
      if (expr.args_[DOC_KEY_IDX]->datum_meta_.type_ == ObJsonType) {
        ObIJsonBase *j_doc_key_base = nullptr;
        bool is_doc_key_null = false;
        if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, temp_allocator, DOC_KEY_IDX, j_doc_key_base, is_doc_key_null))) {
        } else if (is_doc_key_null || OB_ISNULL(j_doc_key_base) || j_doc_key_base->json_type() == ObJsonNodeType::J_NULL) {
          has_doc_key = false;
        } else if (j_doc_key_base->json_type() == ObJsonNodeType::J_STRING) {
          doc_key.assign_ptr(j_doc_key_base->get_data(), static_cast<int32_t>(j_doc_key_base->get_data_length()));
          if (doc_key.empty()) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key is empty");
          } else {
            has_doc_key = true;
          }
        } else {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key must be string type");
        }
      } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, temp_allocator, *arg_doc_key,
                           expr.args_[DOC_KEY_IDX]->datum_meta_, expr.args_[DOC_KEY_IDX]->obj_meta_.has_lob_header(), doc_key))) {
      } else if (doc_key.empty()) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, doc key is empty");
      } else {
        has_doc_key = true;
      }
    }

    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObJsonExprHelper::get_json_doc(expr, ctx, temp_allocator, DOCUMENTS_IDX,
          j_base, is_null_result))) {
    } else if (is_null_result || OB_ISNULL(j_base)) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, document_array is null");
    } else if (j_base->json_type() != ObJsonNodeType::J_ARRAY) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, document_array is not array");
    } else if (OB_ISNULL(document_array = static_cast<ObJsonArray *>(j_base))) {
      ret = OB_ERR_UNEXPECTED;
    } else if (document_array->element_count() == 0) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, document_array is empty");
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && i < document_array->element_count(); i++) {
        ObJsonNode *doc_node = document_array->get_value(i);
        if (OB_ISNULL(doc_node)) {
          ret = OB_INVALID_ARGUMENT;
        } else if (has_doc_key && doc_node->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, document_array element is not object");
        } else if (!has_doc_key && doc_node->json_type() != ObJsonNodeType::J_STRING) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_rerank, document_array element is not string");
        }
      }
    }
    
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(temp_allocator, model_id, info))) {
    } else if (OB_ISNULL(endpoint_resolver)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                   model_id, temp_allocator, resolved_endpoint))) {
    } 

    if (OB_FAIL(ret)) {
    } else if (has_doc_key) {
      if (OB_FAIL(eval_ai_rerank_with_doc_key(expr, ctx, temp_allocator, model_id,
                  query, document_array, doc_key, *info, *endpoint_info, res))) {
      }
    } else {
      if (OB_ISNULL(info)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ObAIFuncUtils::check_info_type_rerank(info))) {
      } else if (OB_FAIL(ObAIFuncUtils::get_header(temp_allocator, *info, *endpoint_info, header_array))) {
      } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(temp_allocator, result_array))) {
      }

      if (OB_SUCC(ret)) {
        ObString score_key(SCORE_KEY);
        int64_t end_idx = 0;
        ObJsonArray *compact_array = nullptr;
        ObJsonArray *batch_result_array = nullptr;
        ObJsonArray *batch_document_array = nullptr;
        for (int64_t i = 0; OB_SUCC(ret) && i < document_array->element_count(); i += batch_size) {
          end_idx = i + batch_size;
          if (end_idx > document_array->element_count()) {
            end_idx = document_array->element_count();
          }
          if (OB_FAIL(construct_batch_document_array(temp_allocator, document_array, i, end_idx, batch_document_array))) {
          } else if (OB_FAIL(inner_eval_ai_rerank(temp_allocator, *info, *endpoint_info, header_array, query, batch_document_array, batch_result_array))) {
          } else if (OB_FAIL(batch_result_add_base(temp_allocator, batch_result_array, i))) {
          } else if (OB_FAIL(compact_json_array_by_key(temp_allocator, result_array, batch_result_array, score_key, compact_array))) {
          } else {
            result_array = compact_array;
          }
        }
      }
      if (OB_SUCC(ret)) {
        ObString raw_str;
        if (OB_FAIL(result_array->get_raw_binary(raw_str, &temp_allocator))) {
        } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, res, raw_str))) {
        }
      }
    }
  }
  return ret;
}

int ObExprAIRerank::eval_ai_rerank_with_doc_key(const ObExpr &expr, ObEvalCtx &ctx, ObIAllocator &allocator, 
                                                ObString& model_id, ObString& query, ObJsonArray *document_array,
                                                ObString& doc_key, const ObAIFuncExprInfo &info,
                                                const share::ObAiModelEndpointInfo &endpoint_info, ObDatum &res)
{
  UNUSED(model_id);
  INIT_SUCC(ret);
  ObJsonArray *doc_array = nullptr;
  ObJsonArray *result_array = nullptr;
  ObJsonArray *sorted_document_array = nullptr;
  ObArray<ObString> header_array;
  const int64_t batch_size = 20;
  if (OB_FAIL(get_doc_array_from_documents_array_with_key(allocator, document_array, doc_key, doc_array))) {
  } else if (OB_FAIL(ObAIFuncUtils::check_info_type_rerank(&info))) {
  } else if (OB_FAIL(ObAIFuncUtils::get_header(allocator, info, endpoint_info, header_array))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result_array))) {
  } else {
    ObString score_key(SCORE_KEY);
    int64_t end_idx = 0;
    ObJsonArray *compact_array = nullptr;
    ObJsonArray *batch_result_array = nullptr;
    ObJsonArray *batch_document_array = nullptr;
    for (int64_t i = 0; OB_SUCC(ret) && i < doc_array->element_count(); i += batch_size) {
      end_idx = i + batch_size;
      if (end_idx > doc_array->element_count()) {
        end_idx = doc_array->element_count();
      }
      if (OB_FAIL(construct_batch_document_array(allocator, doc_array, i, end_idx, batch_document_array))) {
      } else if (OB_FAIL(inner_eval_ai_rerank(allocator, info, endpoint_info, header_array, query, batch_document_array, batch_result_array))) {
      } else if (OB_FAIL(batch_result_add_base(allocator, batch_result_array, i))) {
      } else if (OB_FAIL(compact_json_array_by_key(allocator, result_array, batch_result_array, score_key, compact_array))) {
      } else {
        result_array = compact_array;
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(sort_document_array_by_model_result(allocator, document_array, result_array, sorted_document_array))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    ObString raw_str;
    if (OB_FAIL(sorted_document_array->get_raw_binary(raw_str, &allocator))) {
    } else if (OB_FAIL(ObJsonExprHelper::pack_json_str_res(expr, ctx, res, raw_str))) {
    }
  }
  return ret;
}

int ObExprAIRerank::get_doc_array_from_documents_array_with_key(ObIAllocator &allocator, ObJsonArray *document_object_array, ObString &doc_key, ObJsonArray *&doc_array)
{
  INIT_SUCC(ret);
  if (OB_ISNULL(document_object_array)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (doc_key.empty()) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObJsonArray *doc_array_obj = nullptr;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, doc_array_obj))) {
    } else {
      int64_t count = document_object_array->element_count();
      ObJsonNode *doc_node = nullptr;
      ObJsonNode *doc_value_node = nullptr;
      ObJsonObject *doc_obj = nullptr;
      for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
        doc_node = document_object_array->get_value(i);
        if (OB_ISNULL(doc_node)) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is null");
        } else if (doc_node->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_ARGUMENT;
          LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, document_array element is not object");
        } else {
          doc_obj = static_cast<ObJsonObject *>(doc_node);
          doc_value_node = doc_obj->get_value(doc_key);
          if (OB_ISNULL(doc_value_node)) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is null");
          } else if (doc_value_node->json_type() != ObJsonNodeType::J_STRING) {
            ret = OB_INVALID_ARGUMENT;
            LOG_USER_ERROR (OB_INVALID_ARGUMENT, "ai_rerank, the value for given doc key is not string");
          } else if (OB_FAIL(doc_array_obj->append(doc_value_node))) {
          }
        }
      }
    }
    if (OB_SUCC(ret)) {
      doc_array = doc_array_obj;
    }
  }
  return ret;
}

int ObExprAIRerank::sort_document_array_by_model_result(ObIAllocator &allocator, ObJsonArray *document_array, ObJsonArray *model_result_array, ObJsonArray *&sorted_document_array)
{
  INIT_SUCC(ret);
  ObJsonArray *sorted_document_array_obj = nullptr;
  if (OB_ISNULL(document_array) || OB_ISNULL(model_result_array)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (model_result_array->element_count() != document_array->element_count()) {
    ret = OB_INVALID_DATA;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, sorted_document_array_obj))) {
  } else {
    int64_t count = model_result_array->element_count();
    for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
      ObJsonNode *res_node = model_result_array->get_value(i);
      if (OB_ISNULL(res_node) || res_node->json_type() != ObJsonNodeType::J_OBJECT) {
        ret = OB_INVALID_DATA;
      } else {
        ObJsonObject *model_result_obj = static_cast<ObJsonObject *>(res_node);
        ObJsonNode *index_node = model_result_obj->get_value(INDEX_KEY);
        ObJsonNode *score_node = model_result_obj->get_value(SCORE_KEY);
        int64_t index = -1;
        double score_val = 0.0;
        if (!extract_json_int(index_node, index) || index < 0 || index >= document_array->element_count()
            || !extract_json_score(score_node, score_val)) {
          ret = OB_INVALID_DATA;
        } else {
          ObJsonNode *origin_node = document_array->get_value(index);
          ObJsonObject *new_obj = nullptr;
          ObJsonNode *cloned_score = nullptr;
          if (OB_ISNULL(origin_node) || origin_node->json_type() != ObJsonNodeType::J_OBJECT) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_ISNULL(new_obj = static_cast<ObJsonObject *>(origin_node->clone(&allocator, true)))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_ISNULL(cloned_score = score_node->clone(&allocator, true))) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else if (OB_FAIL(new_obj->add(MODEL_SCORE_KEY, cloned_score))) {
          } else if (OB_FAIL(sorted_document_array_obj->append(new_obj))) {
          }
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    sorted_document_array = sorted_document_array_obj;
  }
  return ret;
}

int ObExprAIRerank::construct_batch_document_array(ObIAllocator &allocator, ObJsonArray *document_array, int64_t start_idx, int64_t end_idx, ObJsonArray *&batch_document_array)
{
  INIT_SUCC(ret);
  ObJsonArray *result_array = nullptr;
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result_array))) {
  } else {
    for (int64_t i = start_idx; OB_SUCC(ret) && i < end_idx; i++) {
      if (OB_FAIL(result_array->append(document_array->get_value(i)))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    batch_document_array = result_array;
  }
  return ret;
}

int ObExprAIRerank::inner_eval_ai_rerank(ObIAllocator &allocator, 
                                          const ObAIFuncExprInfo &info, 
                                          const share::ObAiModelEndpointInfo &endpoint_info,
                                          ObArray<ObString> &header_array, 
                                          ObString &query, 
                                          ObJsonArray *document_array, 
                                          ObJsonArray *&result_array)
{
  INIT_SUCC(ret);
  ObJsonArray *res = nullptr;
  ObJsonObject *config_json = nullptr;
  ObJsonObject *body = nullptr;
  ObJsonObject *http_response = nullptr;
  ObIJsonBase *response = nullptr;
  ObAIFuncClient ai_client;
  if (OB_ISNULL(document_array) || document_array->element_count() <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObAIFuncUtils::get_rerank_body(allocator, info, endpoint_info, query, document_array, config_json, body))) {
  } else if (OB_FAIL(ai_client.send_post(allocator, endpoint_info.get_url(), header_array, body, http_response))) {
  } else if (OB_FAIL(ObAIFuncUtils::parse_rerank_output(allocator, endpoint_info, http_response, response))) {
  } else if (OB_ISNULL(response) || response->json_type() != ObJsonNodeType::J_ARRAY) {
    ret = OB_INVALID_DATA;
  } else if (OB_ISNULL(res = static_cast<ObJsonArray *>(response))) {
    ret = OB_ERR_UNEXPECTED;
  } else if (res->element_count() != document_array->element_count()) {
    ret = OB_INVALID_DATA;
  } else {
    result_array = res;
  }

  if (ret == OB_INVALID_DATA) {
    ObString response_str;
    if (OB_NOT_NULL(http_response) && OB_SUCCESS == ObAIFuncJsonUtils::print_json_to_str(allocator, http_response, response_str)) {
      char http_message_str[1024];
      snprintf(http_message_str, sizeof(http_message_str), "unexpected http message: %s", response_str.ptr());
      ObString ob_http_message_str(http_message_str);
      FORWARD_USER_ERROR(ret, ob_http_message_str.ptr());
    } else {
      FORWARD_USER_ERROR(ret, "unexpected http message");
    }
  }
  return ret;
}

int ObExprAIRerank::batch_result_add_base(ObIAllocator &allocator, ObJsonArray *array, int64_t start_idx)
{
  INIT_SUCC(ret);
  if (OB_ISNULL(array) || array->element_count() <= 0 || start_idx < 0) {
    ret = OB_INVALID_DATA;
  } else {
    int64_t count = array->element_count();
    ObString index_key(INDEX_KEY);
    ObString score_key(SCORE_KEY);
    ObArray<bool> seen;
    for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
      if (OB_FAIL(seen.push_back(false))) {
      }
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < count; i++) {
      ObJsonNode *node = array->get_value(i);
      if (OB_ISNULL(node) || node->json_type() != ObJsonNodeType::J_OBJECT) {
        ret = OB_INVALID_DATA;
      } else {
        ObJsonObject *obj = static_cast<ObJsonObject *>(node);
        ObJsonNode *idx_node = obj->get_value(index_key);
        ObJsonNode *score_node = obj->get_value(score_key);
        int64_t local_idx = -1;
        double score_val = 0.0;
        if (!extract_json_int(idx_node, local_idx)
            || local_idx < 0 || local_idx >= count || seen.at(local_idx)
            || !extract_json_score(score_node, score_val)) {
          ret = OB_INVALID_DATA;
        } else {
          seen.at(local_idx) = true;
          int64_t global_id = local_idx + start_idx;
          if (idx_node->json_type() == ObJsonNodeType::J_INT) {
            static_cast<ObJsonInt *>(idx_node)->set_value(global_id);
            idx_node->update_serialize_size_cascade();
          } else {
            ObJsonInt *new_int_node = nullptr;
            if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, global_id, new_int_node))) {
            } else if (OB_FAIL(obj->replace(idx_node, new_int_node))) {
            }
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIRerank::compact_json_array_by_key(ObIAllocator &allocator, ObJsonArray *array1, ObJsonArray *array2, ObString &score_key, ObJsonArray *&compact_array)
{
  INIT_SUCC(ret);
  ObJsonArray *result_array = nullptr;
  if (OB_ISNULL(array1) || OB_ISNULL(array2)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_array(allocator, result_array))) {
  } else {
    ObString index_key(INDEX_KEY);
    ObArray<RerankSortableItem> items;
    int64_t order_pos = 0;
    ObJsonArray *arrays[2] = {array1, array2};
    for (int64_t a_idx = 0; OB_SUCC(ret) && a_idx < 2; ++a_idx) {
      ObJsonArray *cur_arr = arrays[a_idx];
      int64_t count = cur_arr->element_count();
      for (int64_t i = 0; OB_SUCC(ret) && i < count; ++i) {
        ObJsonNode *node = cur_arr->get_value(i);
        if (OB_ISNULL(node) || node->json_type() != ObJsonNodeType::J_OBJECT) {
          ret = OB_INVALID_DATA;
        } else {
          ObJsonObject *obj = static_cast<ObJsonObject *>(node);
          ObJsonNode *score_node = obj->get_value(score_key);
          double score = 0.0;
          if (!extract_json_score(score_node, score)) {
            ret = OB_INVALID_DATA;
          } else {
            ObJsonNode *idx_node = obj->get_value(index_key);
            int64_t doc_idx = -1;
            bool has_index = extract_json_int(idx_node, doc_idx);
            if (OB_FAIL(items.push_back(RerankSortableItem(obj, score, doc_idx, has_index, order_pos++)))) {
            }
          }
        }
      }
    }
    if (OB_SUCC(ret) && items.count() > 0) {
      RerankItemComparator cmp;
      lib::ob_sort(&items.at(0), &items.at(0) + items.count(), cmp);
      for (int64_t i = 0; OB_SUCC(ret) && i < items.count(); ++i) {
        if (OB_FAIL(result_array->append(items.at(i).obj_))) {
        }
      }
    }
  }
  if (OB_SUCC(ret)) {
    compact_array = result_array;
  }
  return ret;
}

int ObExprAIRerank::construct_config_json(ObIAllocator &allocator, int64_t top_n, int64_t return_doc, ObJsonObject *&config_json)
{
  INIT_SUCC(ret);
  ObJsonObject *config_obj = nullptr;
  ObJsonInt *top_n_obj = nullptr;
  ObJsonBoolean *return_doc_obj = nullptr;
  bool return_doc_value = true;
  if (return_doc <= 0) {
    return_doc_value = false;
  }
  if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config_obj))) {
  } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_boolean(allocator, return_doc_value, return_doc_obj))) {
  } else if (OB_FAIL(config_obj->add("return_documents", return_doc_obj))) {
  } else if (top_n > 0) {
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, top_n, top_n_obj))) {
    } else if (OB_FAIL(config_obj->add("top_n", top_n_obj))) {
    }
  }
  if (OB_SUCC(ret)) {
    config_json = config_obj;
  }
  return ret;
}

int ObExprAIRerank::cg_expr(ObExprCGCtx &expr_cg_ctx, 
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
  //   bool got_data = false;
  //   ObObj const_data;
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
  //   } else {
  //     ObAIFuncExprInfo *info = OB_NEWx(ObAIFuncExprInfo, allocator, *allocator, T_FUN_SYS_AI_RERANK);
  //     if (OB_ISNULL(info)) {
  //       ret = OB_ALLOCATE_MEMORY_FAILED;
  //       LOG_WARN("failed to allocate memory for ai rerank info", K(ret));
  //     } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //       LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //     } else {
  //       rt_expr.extra_info_ = info;
  //     }
  //   }
  // }
  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIRerank::eval_ai_rerank;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase

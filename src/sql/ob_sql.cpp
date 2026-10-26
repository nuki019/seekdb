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

#define USING_LOG_PREFIX SQL
#include "config_bridge.h"
#include "sql/ob_sql.h"
#include "lib/json/ob_json_print_utils.h"
#include "lib/stat/ob_diagnostic_info_guard.h"
#include "query/runtime/ob_query_runtime_environment.h"
#include "share/ob_truncated_string.h"
#include "sql/plan_cache/ob_ps_cache.h"
#include "sql/plan_cache/ob_pcv_set.h"
#include "sql/ob_sql_init.h"
#include "sql/resolver/ob_resolver.h"
#include "sql/resolver/cmd/ob_variable_set_stmt.h"
#include "sql/resolver/cmd/ob_call_procedure_stmt.h"
#include "sql/resolver/cmd/ob_anonymous_block_stmt.h"
#include "sql/privilege_check/ob_privilege_check.h"
#include "sql/rewrite/ob_transformer_impl.h"
#include "sql/rewrite/ob_transform_pre_process.h"
#include "sql/code_generator/ob_code_generator.h"
#include "sql/plan_cache/ob_values_table_compression.h"
#include "sql/pl/ob_pl_resolver.h"
#include "lib/utility/ob_smart_call.h"
#include "sql/monitor/show_trace/ob_show_trace.h"
#include "sql/optimizer/ob_optimizer.h"
#include "share/rc/ob_server_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cmath>
#include <cerrno>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#ifdef _WIN32
#include <io.h>
#else
#include <glob.h>
#endif

namespace oceanbase
{
using namespace common;
using namespace rpc::frame;
using namespace obcall;
using namespace share;
using namespace share::schema;
using namespace pl;

namespace sql
{

const int64_t ObSql::max_error_length = 80;
const int64_t ObSql::SQL_MEM_SIZE_LIMIT = 1024 * 1024 * 64;

int ObSql::init(common::ObOptStatManager *opt_stat_mgr,
                common::ObITabletScan *vt_partition_service,
                common::ObAddr &addr,
                ObPlanCache &plan_cache,
                ObPsCache &ps_cache,
                pl::ObPL &pl_engine,
                query::ObIPlanCacheAccessService &plan_cache_access_service,
                query::ObIQueryRuntimeEnvironment &query_runtime_environment,
                query::ObIRootCommandService &root_command_service,
                query::ObILocalCommandService &local_command_service,
                query::ObIChangeStreamService &change_stream_service,
                query::ObIDdlExecutionLimiter &ddl_execution_limiter,
                ObIVirtualTableFactoryProvider &virtual_table_factory_provider,
                common::ObISrsProvider &srs_provider,
                common::ObILobReadService &lob_read_service)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(opt_stat_mgr)
      || OB_ISNULL(vt_partition_service)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    if (OB_FAIL(queue_.init(1, 512))) {
    } else {
      queue_.bind_root_command_service(root_command_service);
      opt_stat_mgr_ = opt_stat_mgr;
      vt_partition_service_ = vt_partition_service;
      plan_cache_ = &plan_cache;
      ps_cache_ = &ps_cache;
      pl_engine_ = &pl_engine;
      plan_cache_access_service_ = &plan_cache_access_service;
      query_runtime_environment_ = &query_runtime_environment;
      root_command_service_ = &root_command_service;
      local_command_service_ = &local_command_service;
      change_stream_service_ = &change_stream_service;
      ddl_execution_limiter_ = &ddl_execution_limiter;
      virtual_table_factory_provider_ = &virtual_table_factory_provider;
      srs_provider_ = &srs_provider;
      lob_read_service_ = &lob_read_service;
      self_addr_ = addr;
      inited_ = true;
      queue_.start();
    }
  }
  return ret;
}

void ObSql::destroy() {
  if (inited_) {
    queue_.destroy();
    plan_cache_ = NULL;
    ps_cache_ = NULL;
    pl_engine_ = NULL;
    plan_cache_access_service_ = NULL;
    query_runtime_environment_ = NULL;
    root_command_service_ = NULL;
    local_command_service_ = NULL;
    change_stream_service_ = NULL;
    ddl_execution_limiter_ = NULL;
    virtual_table_factory_provider_ = NULL;
    srs_provider_ = NULL;
    lob_read_service_ = NULL;
    inited_ = false;
  }
}

void ObSql::bind_resolver_runtime_services(ObResolverParams &resolver_ctx)
{
  resolver_ctx.plan_cache_ = plan_cache_;
  resolver_ctx.pl_sql_runtime_ = this;
  resolver_ctx.pl_engine_ = pl_engine_;
  resolver_ctx.dependency_info_queue_ = &queue_;
  resolver_ctx.root_command_service_ = root_command_service_;
  resolver_ctx.srs_provider_ = srs_provider_;
  resolver_ctx.lob_read_service_ = lob_read_service_;
}

void ObSql::bind_exec_context_runtime_services(ObExecContext &exec_ctx)
{
  ObExecContext::RuntimeServices services = exec_ctx.get_runtime_services();
  services.lob_read_service_ = lob_read_service_;
  services.plan_cache_ = plan_cache_;
  services.ps_cache_ = ps_cache_;
  services.plan_cache_access_service_ = plan_cache_access_service_;
  services.pl_sql_runtime_ = this;
  services.pl_engine_ = pl_engine_;
  services.prepared_statement_runtime_ = this;
  services.sql_execution_id_provider_ = this;
  services.query_runtime_environment_ = query_runtime_environment_;
  services.root_command_service_ = root_command_service_;
  services.local_command_service_ = local_command_service_;
  services.change_stream_service_ = change_stream_service_;
  services.ddl_execution_limiter_ = ddl_execution_limiter_;
  services.virtual_table_factory_provider_ = virtual_table_factory_provider_;
  services.srs_provider_ = srs_provider_;
  exec_ctx.set_runtime_services(services);
}


int ObSql::stmt_prepare(const common::ObString &stmt,
                        ObSqlCtx &context,
                        ObResultSet &result,
                        bool is_inner_sql/*true*/)
{
  int ret = OB_SUCCESS;
  LinkExecCtxGuard link_guard(result.get_session(), result.get_exec_context());
  if (OB_FAIL(sanity_check(context))) {
  } else if (OB_FAIL(handle_ps_prepare(stmt, context, result, is_inner_sql))) {
  }
  if (OB_FAIL(ret) && OB_SUCCESS == result.get_errcode()) {
    result.set_errcode(ret);
  }
  if (OB_FAIL(ret)) {
    rollback_implicit_trans_when_fail(result, ret);
  }
  return ret;
}

int ObSql::stmt_query(const common::ObString &stmt, ObSqlCtx &context, ObResultSet &result)
{
  int ret = OB_SUCCESS;
  LinkExecCtxGuard link_guard(result.get_session(), result.get_exec_context());
  ObTraceSpanGuard compile_span(&result.get_session(), TRACE_SQL_COMPILE);
  ObTruncatedString trunc_stmt(stmt);
#ifndef NDEBUG
  LOG_INFO("Begin to handle text statement",
           "sess_id", result.get_session().get_server_sid(),
           "execution_id", result.get_session().get_current_execution_id(),
           K(trunc_stmt));
#endif
  NG_TRACE(parse_begin);
  //1 check inited
  if (OB_FAIL(sanity_check(context))) {
  } else if (OB_FAIL(handle_text_query(stmt, context, result))) {
    if (OB_EAGAIN != ret) {
    }
  }
  //LOG_DEBUG("result errno", N_ERR_CODE, result.get_errcode(), K(ret));
  if (OB_SUCCESS != ret
      && OB_SUCCESS == result.get_errcode()) {
    result.set_errcode(ret);
  }
  if (OB_ISNULL(result.get_physical_plan())) {
  } else {
  }
  NG_TRACE_EXT(stmt_query_end, OB_ID(stmt),
               context.is_sensitive_ ? ObString(OB_MASKED_STR) : trunc_stmt.string(),
               OB_ID(stmt_len), stmt.length());

  if (OB_FAIL(ret)) {
    rollback_implicit_trans_when_fail(result, ret);
  }
  return ret;
}

int ObSql::stmt_execute(const ObPsStmtId stmt_id,
                        const stmt::StmtType stmt_type,
                        const ParamStore &params,
                        ObSqlCtx &context,
                        ObResultSet &result,
                        bool is_inner_sql)
{
  int ret = OB_SUCCESS;
  LinkExecCtxGuard link_guard(result.get_session(), result.get_exec_context());
  if (OB_FAIL(sanity_check(context))) {
  } else if (OB_FAIL(init_result_set(context, result))) {
  } else if (OB_FAIL(handle_ps_execute(stmt_id, stmt_type, params, context, result, is_inner_sql))) {
  }
  if (OB_FAIL(ret) && OB_SUCCESS == result.get_errcode()) {
    result.set_errcode(ret);
  }
  if (OB_FAIL(ret)) {
    rollback_implicit_trans_when_fail(result, ret);
  }
  return ret;
}


int ObSql::fill_result_set(ObResultSet &result_set,
                           ObSqlCtx *context,
                           const PlanCacheMode mode,
                           ObStmt &basic_stmt)
{
  int ret = OB_SUCCESS;
  ObStmt *stmt = &basic_stmt;
  ObExecContext &ectx = result_set.get_exec_context();
  ObPhysicalPlanCtx *pctx = ectx.get_physical_plan_ctx();
  if (OB_UNLIKELY(NULL == context) || OB_UNLIKELY(NULL == context->session_info_) || OB_UNLIKELY(NULL == pctx)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    result_set.set_affected_rows(0);
    result_set.set_warning_count(0);
    result_set.set_message("");
    ObString type_name = ObString::make_string("varchar");
    number::ObNumber number;
    number.set_zero();
    ObField field;
    common::ObIAllocator &alloc = result_set.get_mem_pool();
    ObCharsetType result_charset = CHARSET_INVALID;
    ObCollationType collation_type = CS_TYPE_INVALID;
    context->session_info_->get_character_set_results(result_charset);
    collation_type =
        ObCharset::get_default_collation_by_mode(result_charset);
    switch (stmt->get_stmt_type()) {
    case stmt::T_SELECT: {
      if (OB_FAIL(fill_select_result_set(result_set, context, mode, collation_type, type_name,
                                         basic_stmt, field))) {
      }
      break;
    }
    case stmt::T_INSERT:
    case stmt::T_REPLACE:
    case stmt::T_UPDATE:
    case stmt::T_DELETE: {
      break;
    }
    case stmt::T_EXPLAIN: {
      ObString tname = ObString::make_string("explain_table");
      ObString cname = ObString::make_string("Query Plan");
      field.tname_ = tname;
      field.org_tname_ = tname;
      field.cname_ = cname;
      field.org_cname_ = cname;
      field.type_.set_type(ObVarcharType);
      field.charsetnr_ = CS_TYPE_UTF8MB4_GENERAL_CI;
      field.type_.set_collation_type(CS_TYPE_UTF8MB4_GENERAL_CI);
      field.type_.set_collation_level(CS_LEVEL_IMPLICIT);
      field.type_.set_varchar(type_name);
      if (OB_FAIL(result_set.reserve_field_columns(1))) {
      } else if (OB_FAIL(result_set.add_field_column(field))) {
      }
      break;
    }
    case stmt::T_CALL_PROCEDURE: {
      ObCallProcedureStmt &call_stmt = static_cast<ObCallProcedureStmt&>(basic_stmt);
      ObString tname = ObString::make_string("procedure");
      if (NULL == call_stmt.get_call_proc_info()) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        int64_t size = call_stmt.get_call_proc_info()->get_output_count();
        field.charsetnr_ = CS_TYPE_UTF8MB4_GENERAL_CI;

        if (0 == size) {
          break;
        }

        if (OB_SUCC(ret) && OB_FAIL(result_set.reserve_field_columns(size))) {
        }
        for (int64_t i = 0; OB_SUCC(ret) && i < size; ++i) {
          if (!call_stmt.get_call_proc_info()->is_client_out_param_by_out_param_id(i)) {
            // only return client out param to client
            continue;
          }
          ObCollationType charsetnr;
          const ObDataType *type = call_stmt.get_call_proc_info()->get_out_type().at(i).get_data_type();
          ObObjType out_obj_type = call_stmt.get_call_proc_info()->get_out_type().at(i).get_obj_type();
          if (ObUnknownType == out_obj_type
            || ObExtendType == out_obj_type) {
            // do nothing ...
          } else if (OB_NOT_NULL(type)) {
            if (ob_is_string_or_lob_type(out_obj_type)
                  && CS_TYPE_ANY == type->get_collation_type()) {
              charsetnr = ObCharset::is_valid_collation(collation_type)
                            ? collation_type
                            : CS_TYPE_UTF8MB4_BIN;
            } else {
              OZ (ObCharset::get_default_collation(type->get_collation_type(), charsetnr));
            }
            OX (field.charsetnr_ = static_cast<uint16_t>(charsetnr));
          }
          if (OB_SUCC(ret)) {
            field.type_.set_type(out_obj_type);
            if (OB_NOT_NULL(type)) {
              field.accuracy_ = type->get_accuracy();
              // Setup Collation and Collation levl
              if (ob_is_string_or_lob_type(out_obj_type)
                  || ob_is_enum_or_set_type(out_obj_type)) {
                field.type_.set_collation_type(type->get_collation_type());
                field.type_.set_collation_level(type->get_collation_level());
              }
            }
            if (ObExtendType == out_obj_type) {
              field.length_ = field.accuracy_.get_length();
            } else if (ObCharType == out_obj_type
                      || ObVarcharType == out_obj_type) {
              if (-1 == field.accuracy_.get_length()) {
                field.length_ = ObCharType == out_obj_type ?
                  OB_MAX_EXTENDED_CHAR_LENGTH_BYTE : OB_MAX_EXTENDED_VARCHAR_LENGTH;
              } else {
                field.length_ = field.accuracy_.get_length();
              }
            } else if (ob_is_enum_or_set_type(out_obj_type)) {
              CK (OB_NOT_NULL(type));
              OZ (common::ObField::get_field_mb_length(field.type_.get_type(),
                                                      field.accuracy_,
                                                      type->get_collation_type(),
                                                      field.length_));
              OX (field.type_.set_type(ObVarcharType));
            } else {
              OZ (common::ObField::get_field_mb_length(field.type_.get_type(),
                                                      field.accuracy_,
                                                      common::CS_TYPE_INVALID,
                                                      field.length_));
            }
            // Setup Scale
            if (OB_SUCC(ret)) {
              if (ObVarcharType == field.type_.get_type()) {
                field.type_.set_varchar(type_name);
              } else if (ObNumberType == field.type_.get_type()) {
                field.type_.set_number(number);
              }
            }
          }
          if (OB_SUCC(ret)) {
            if (OB_FAIL(ob_write_string(alloc, tname, field.tname_))) {
            } else if (OB_FAIL(ob_write_string(alloc, tname, field.org_tname_))) {
            } else if (OB_FAIL(ob_write_string(alloc, call_stmt.get_call_proc_info()->get_out_name().at(i), field.cname_))) {
            } else if (OB_FAIL(ob_write_string(alloc, call_stmt.get_call_proc_info()->get_out_name().at(i), field.org_cname_))) {
            } else if (OB_FAIL(ob_write_string(alloc, call_stmt.get_call_proc_info()->get_out_type_name().at(i), field.type_name_))) {
            } else if (OB_FAIL(ob_write_string(alloc, call_stmt.get_call_proc_info()->get_out_type_owner().at(i), field.type_owner_))) {
            } else { /*do nothing*/ }
          }

          if (OB_SUCC(ret)) {
            if (OB_FAIL(result_set.add_field_column(field))) {
            }
          }

          if (OB_SUCC(ret)) {
            field.cname_.assign(NULL, 0);
            field.org_cname_.assign(NULL, 0);
            field.dname_.assign(NULL, 0);
            field.tname_.assign(NULL, 0);
            field.org_tname_.assign(NULL, 0);
            field.type_.reset();
            field.type_.set_type(ObExtendType);
            field.type_name_.assign(NULL, 0);
            field.type_owner_.assign(NULL, 0);
          }
        }
        break;
      }
    }
    default:
      break;
    }

    const int64_t question_marks_count = pctx->get_ps_fixed_array_index() != nullptr && stmt::T_ANONYMOUS_BLOCK != stmt->get_stmt_type() ?
          pctx->get_orig_question_mark_cnt() : stmt->get_query_ctx()->get_prepare_param_count();
    // param column is only needed in ps mode
    if (OB_SUCC(ret) && question_marks_count > 0
        && (PC_PS_MODE == mode || PC_PL_MODE == mode)) {
      if (OB_FAIL(result_set.reserve_param_columns(question_marks_count))) {
      }
      ObAnonymousBlockStmt *anonymous_stmt = NULL;
      ObCallProcedureStmt *call_stmt = NULL;
      if (stmt::T_ANONYMOUS_BLOCK == stmt->get_stmt_type()) {
        CK (OB_NOT_NULL(anonymous_stmt = static_cast<ObAnonymousBlockStmt *>(stmt)));
        OZ (result_set.reserve_field_columns(anonymous_stmt->get_out_idx().num_members()));
      } else if (stmt::T_CALL_PROCEDURE == stmt->get_stmt_type()) {
        CK (OB_NOT_NULL(call_stmt = static_cast<ObCallProcedureStmt *>(stmt)));
      }
      for (int64_t i = 0; OB_SUCC(ret) && i < question_marks_count; ++i) {
        ObField param_field;
        param_field.type_.set_type(ObIntType); // @bug
        param_field.cname_ = ObString::make_string("?");
        if (OB_NOT_NULL(pctx->get_ps_fixed_array_index()) &&
            OB_NOT_NULL(anonymous_stmt) &&
            is_exist_in_fixed_param_idx(i, *pctx->get_ps_fixed_array_index())) {
          continue;
        } else if (OB_NOT_NULL(anonymous_stmt) && anonymous_stmt->get_out_idx().has_member(i)) {
          ObField column_field;
          column_field.type_.set_type(ObNullType);
          OX (param_field.inout_mode_ = ObRoutineParamInOut::SP_PARAM_INOUT);
          OZ (result_set.add_field_column(column_field),
              K(i), K(question_marks_count), K(column_field));
        } else if (OB_NOT_NULL(call_stmt) && call_stmt->get_call_proc_info()->is_out_param(i)) {
          OX (param_field.inout_mode_ = ObRoutineParamInOut::SP_PARAM_INOUT);
        }
        OZ (result_set.add_param_column(param_field),
            K(param_field), K(i), K(question_marks_count));
      }
    }
  }
  return ret;
}

int ObSql::fill_select_result_set(ObResultSet &result_set, ObSqlCtx *context, const PlanCacheMode mode,
                                  ObCollationType collation_type, const ObString &type_name,
                                  ObStmt &basic_stmt, ObField &field)
{
  int ret = OB_SUCCESS;
  ObSelectStmt *select_stmt = static_cast<ObSelectStmt *>(&basic_stmt);
  if (select_stmt->has_select_into()) { //  for select into, no rows return
    // do nothing.
  } else {
    int64_t size = select_stmt->get_select_item_size();
    common::ObIAllocator &alloc = result_set.get_mem_pool();
    number::ObNumber number;
    number.set_zero();
    if (OB_FAIL(result_set.reserve_field_columns(size))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < size; ++i) {
      const SelectItem &select_item = select_stmt->get_select_item(i);
      ObRawExpr *expr = select_item.expr_;
      bool is_ext_field = false;
      ObSqlString composite_field_name;
      if (OB_UNLIKELY(NULL == expr)) {
        ret = OB_ERR_ILLEGAL_ID;
      } else {
        if (ob_is_string_or_lob_type(expr->get_data_type())
            && CS_TYPE_BINARY != expr->get_collation_type()
            && ObCharset::is_valid_collation(collation_type)) {
          field.charsetnr_ = static_cast<uint16_t>(collation_type);
        } else {
          field.charsetnr_ = static_cast<uint16_t>(expr->get_collation_type());
        }
      }

      if (OB_SUCC(ret) && expr->get_result_type().is_ext()) {
        is_ext_field = true;
        field.type_.set_collation_type(CS_TYPE_BINARY);
        field.type_.set_collation_level(CS_LEVEL_IMPLICIT);
        field.length_ = OB_MAX_LONGTEXT_LENGTH;
        if (CS_TYPE_BINARY != expr->get_collation_type()
          && ObCharset::is_valid_collation(collation_type)) {
          field.charsetnr_ = static_cast<uint16_t>(collation_type);
        } else {
          field.charsetnr_ = static_cast<uint16_t>(expr->get_collation_type());
        }
      }

      if (OB_SUCC(ret)) {
        // Setup field Type and Accuracy
        field.type_.set_type(expr->get_data_type());
        field.accuracy_ = expr->get_accuracy();
        field.flags_ = static_cast<uint16_t>(expr->get_result_flag());
        // Setup Collation and Collation level
        if (ob_is_string_or_lob_type(static_cast<ObObjType>(expr->get_data_type()))
            || ob_is_enum_or_set_type(static_cast<ObObjType>(expr->get_data_type()))) {
          field.type_.set_collation_type(expr->get_collation_type());
          field.type_.set_collation_level(expr->get_collation_level());
        }
        // Setup Scale
        if (ObVarcharType == field.type_.get_type()) {
          field.type_.set_varchar(type_name);
        } else if (ObNumberType == field.type_.get_type()) {
          field.type_.set_number(number);
        }
        if (context->session_info_->is_varparams_sql_prepare()) {
          // question mark expr has no valid result type in prepare stage
          if (!expr->get_result_type().is_ext()
              && !expr->get_result_type().is_collection_sql_type()
              && !expr->get_result_type().is_enum_or_set()
              && !expr->get_result_type().is_enumset_inner_type()
              && OB_FAIL(expr->get_length_for_meta_in_bytes(
                    field.length_, static_cast<ObCollationType>(field.charsetnr_)))) {
          }
        } else if (expr->get_result_type().is_collection_sql_type()) {
          uint16_t subschema_id = expr->get_result_type().get_subschema_id();
          uint16_t tmp_subschema_id = ObInvalidSqlType;
          uint64_t udt_id = expr->get_result_type().get_udt_id();
          ObSqlUDTMeta udt_meta;
          if (expr->get_result_type().is_collection_sql_type()
              && !ObObjUDTUtil::ob_is_supported_sql_udt(udt_id)) {
            // array type
            field.type_.set_subschema_id(subschema_id);
            field.charsetnr_ = CS_TYPE_UTF8MB4_BIN;
            field.length_ = OB_MAX_LONGTEXT_LENGTH;
          } else if (OB_FAIL(result_set.get_exec_context().get_subschema_id_by_udt_id(udt_id, tmp_subschema_id))) {
          } else if (OB_FAIL(result_set.get_exec_context().get_sqludt_meta_by_subschema_id(tmp_subschema_id, udt_meta))) {
          } else if(ObObjUDTUtil::ob_is_supported_sql_udt(udt_meta.udt_id_)) {
            if (udt_id != udt_meta.udt_id_) {
              ret = OB_ERR_UNEXPECTED;
            }
            field.type_.set_subschema_id(tmp_subschema_id);
            field.charsetnr_ = CS_TYPE_BINARY;
            field.length_ = OB_MAX_LONGTEXT_LENGTH;
            if (OB_SUCC(ret)) {
              if (OB_FAIL(ob_write_string(alloc, ObString(udt_meta.udt_name_len_, udt_meta.udt_name_), field.type_name_))) {
              }
            }
          } else {
            ret = OB_NOT_SUPPORTED;
          }
        } else if (expr->get_result_type().is_ext()) {
          field.type_.meta_.set_extend_type(expr->get_result_type().get_extend_type());
          if (OB_INVALID_ID != expr->get_result_type().get_udt_id()
              && (PL_VARRAY_TYPE == expr->get_result_type().get_extend_type()
                  || PL_NESTED_TABLE_TYPE == expr->get_result_type().get_extend_type()
                  || PL_ASSOCIATIVE_ARRAY_TYPE == expr->get_result_type().get_extend_type()
                  || PL_RECORD_TYPE == expr->get_result_type().get_extend_type())) {
            if (PC_TEXT_MODE == mode
                && NULL == context->secondary_namespace_
                && NULL == context->session_info_->get_pl_context()) {
              // Text protocol convert extend type field to varchar
              field.type_.set_varchar(type_name);
            }
          }
        } else if (!expr->get_result_type().is_ext()
                   && OB_FAIL(expr->get_length_for_meta_in_bytes(
                        field.length_, static_cast<ObCollationType>(field.charsetnr_)))) {
        }
      }
      // SELECT ITEM's alias name and expr name rule examples:
      // SELECT field1+field2 AS f1, field1+3, "thanks", field2 AS f2, field3, "hello" as f4, field1+4 as f5 FROM t1
      // "is_alias":true,  "alias_name":"f1", "expr_name":"f1"
      // "is_alias":false, "alias_name":"", "expr_name":"field1+3"
      // "is_alias":false, "alias_name":"", "expr_name":", "thanks"",
      // "is_alias":true,  "alias_name":"f2", "expr_name":"f2",  "column_name":"field2",
      // "is_alias":true,  "alias_name":"field3", "expr_name":"field3",  "column_name":"field3",
      // "is_alias":true,  "alias_name":"f4", "expr_name":"f4"
      // "is_alias":true,  "alias_name":"f5", "expr_name":"f5"
      //
      ObCollationType field_names_collation =
            ObCharset::is_valid_collation(collation_type) ? collation_type : CS_TYPE_UTF8MB4_BIN;
      if (OB_SUCC(ret)) {
        ObSqlString field_name;
        if (composite_field_name.length() > 0) {
          // need record member name
          if (OB_FAIL(field_name.append(select_item.alias_name_))) {
          } else if (OB_FAIL(field_name.append(composite_field_name.string()))) {
          }
        }
        if (OB_FAIL(ret)) {
          // do nothing
        } else if (OB_FAIL(ObSQLUtils::copy_and_convert_string_charset(alloc,
                                field_name.length() > 0 ? field_name.string() : select_item.alias_name_,
                                field.cname_, CS_TYPE_UTF8MB4_BIN, field_names_collation))) {
        }
      }
      if (OB_SUCC(ret)) {
        bool is_contain_column_ref = false;
        if (expr->is_column_ref_expr()
            || T_FUN_SET_TO_STR == expr->get_expr_type()
            || T_FUN_ENUM_TO_STR == expr->get_expr_type()) {
          const TableItem *table_item = NULL;
          ObString column_name;
          ObColumnRefRawExpr *column_expr = NULL;
          if (OB_FAIL(ObRawExprUtils::get_col_ref_expr_recursively(expr, column_expr))) {
          } else if (OB_NOT_NULL(column_expr)) {
            is_contain_column_ref = true;
            uint64_t table_id = column_expr->get_table_id();
            uint64_t column_id = column_expr->get_column_id();
            ColumnItem *column_item = select_stmt->get_column_item_by_id(table_id, column_id);
            if (OB_ISNULL(column_item)) {
              ret = OB_ERR_ILLEGAL_ID;
            } else {
              column_name = column_item->column_name_;
              table_item = select_stmt->get_table_item_by_id(table_id);
              if (column_expr->is_unique_key_column()) {
                field.flags_ |= UNIQUE_KEY_FLAG;
              }
              if (column_expr->is_mul_key_column()) {
                field.flags_ |= MULTIPLE_KEY_FLAG;
              }
              if (::oceanbase::share::schema::is_heap_table_primary_key_column(column_expr->get_column_flags())) {
                field.flags_ |= PRI_KEY_FLAG;
              }
            }
          }
          if (OB_SUCC(ret) && is_contain_column_ref) {
            if (OB_ISNULL(table_item)) {
              ret = OB_ERR_ILLEGAL_ID;
            } else if (OB_FAIL(ob_write_string(alloc,
                                                column_name,
                                                field.org_cname_))) {
            } else if (OB_FAIL(ObSQLUtils::copy_and_convert_string_charset(
                                    alloc, table_item->database_name_, field.dname_,
                                    CS_TYPE_UTF8MB4_BIN, field_names_collation))) {
            } else if (table_item->alias_name_.length() > 0) {
              if (OB_FAIL(ObSQLUtils::copy_and_convert_string_charset(
                                    alloc, table_item->alias_name_, field.tname_,
                                    CS_TYPE_UTF8MB4_BIN, field_names_collation))) {
              }
            } else {
              if (OB_FAIL(ObSQLUtils::copy_and_convert_string_charset(
                                    alloc, table_item->table_name_, field.tname_,
                                    CS_TYPE_UTF8MB4_BIN, field_names_collation))) {
              }
            }
            if (OB_FAIL(ret)) {
            } else if (OB_FAIL(ob_write_string(alloc, table_item->table_name_, field.org_tname_))) {
            }
          }
        }
      }
      if (OB_SUCC(ret) && !(PC_PS_MODE == mode || PC_PL_MODE == mode)) {
        void *buf = NULL;
        if (OB_ISNULL(buf = alloc.alloc(sizeof(ObParamedSelectItemCtx)))) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          ObSqlString paramed_field_name;
          field.paramed_ctx_ = new(buf) ObParamedSelectItemCtx();
          if (composite_field_name.length() > 0) {
            // need record member name
            if (OB_FAIL(paramed_field_name.append(select_item.paramed_alias_name_))) {
            } else if (OB_FAIL(paramed_field_name.append(composite_field_name.string()))) {
            }
          }
          if (OB_FAIL(ret)) {
            // do nothing
          } else if (OB_FAIL(ob_write_string(alloc,
                                             paramed_field_name.length() > 0
                                               ? paramed_field_name.string()
                                               : select_item.paramed_alias_name_,
                                             field.paramed_ctx_->paramed_cname_))) {
          } else if (OB_FAIL(field.paramed_ctx_->param_str_offsets_.assign(
                                                  select_item.questions_pos_))) {
          } else if (OB_FAIL(field.paramed_ctx_->param_idxs_.assign(select_item.params_idx_))) {
          } else {
            field.paramed_ctx_->neg_param_idxs_ = select_item.neg_param_idx_;
            field.paramed_ctx_->esc_str_flag_ = select_item.esc_str_flag_;
            field.paramed_ctx_->need_check_dup_name_ = select_item.need_check_dup_name_;
            // If the projection column is a column
            field.paramed_ctx_->is_column_field_ = (T_REF_COLUMN == expr->get_expr_type());
            field.is_paramed_select_item_ = true;
          }
        }
      }
      if (OB_FAIL(ret)) {
        // do nothing
      } else if (OB_FAIL(result_set.add_field_column(field))) {
      } else {
        field.cname_.assign(NULL, 0);
        field.org_cname_.assign(NULL, 0);
        field.dname_.assign(NULL, 0);
        field.tname_.assign(NULL, 0);
        field.org_tname_.assign(NULL, 0);
        field.type_.reset();
        field.type_.set_type(ObExtendType);
        field.paramed_ctx_ = NULL;
      }
    }
  }
  return ret;
}

int ObSql::fill_result_set(const ObPsStmtId stmt_id, const ObPsStmtInfo &stmt_info, ObResultSet &result)
{
  int ret = OB_SUCCESS;
  result.set_statement_id(stmt_id);
  result.set_stmt_type(stmt_info.get_stmt_type());
  result.set_literal_stmt_type(stmt_info.get_literal_stmt_type());
  const ObPsSqlMeta &sql_meta = stmt_info.get_ps_sql_meta();
  result.set_p_param_fileds(const_cast<common::ParamsFieldIArray *>(&sql_meta.get_param_fields()));
  result.set_p_column_fileds(const_cast<common::ParamsFieldIArray *>(&sql_meta.get_column_fields()));
  //ObPsSqlMeta::const_column_iterator column_iter = sql_meta.column_begin();
  //result.reserve_field_columns(sql_meta.get_column_size());
  //for (; OB_SUCC(ret) && column_iter != sql_meta.column_end(); ++column_iter) {
    //if (OB_ISNULL(column_iter) || OB_ISNULL(*column_iter)) {
      //ret = OB_ERR_UNEXPECTED;
      //LOG_WARN("column iter is null", K(ret), K(column_iter));
    //} else if (OB_FAIL(result.add_field_column(**column_iter))) {
      //LOG_WARN("add column field failed", K(ret));
    //}
  //}

  //ObPsSqlMeta::const_param_iterator param_iter = sql_meta.param_begin();
  //for (; OB_SUCC(ret) && param_iter != sql_meta.param_end(); ++param_iter) {
    //if (OB_ISNULL(param_iter) || OB_ISNULL(param_iter)) {
      //ret = OB_ERR_UNEXPECTED;
      //LOG_WARN("param iter is null", K(ret), K(param_iter));
    //} else if (OB_FAIL(result.add_param_column(**param_iter))) {
      //LOG_WARN("add param field faield", K(ret));
    //}
  //}
  return ret;
}

int ObSql::do_add_ps_cache(const PsCacheInfoCtx &info_ctx,
                           ObSchemaGetterGuard &schema_guard,
                           ObResultSet &result)
{
  int ret = OB_SUCCESS;
  bool is_contain_tmp_tbl = false;
  ObSQLSessionInfo &session = result.get_session();
  ObPsCache *ps_cache = ps_cache_;
  uint64_t db_id = OB_INVALID_ID;
  (void)session.get_database_id(db_id);
  if (OB_ISNULL(ps_cache)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (      OB_FAIL(check_contain_temporary_table(schema_guard, result, is_contain_tmp_tbl))) {
  } else {
    ObPsStmtItem *ps_stmt_item = NULL;
    ObPsStmtInfo *ref_stmt_info = NULL;
    bool duplicate_prepare = false;
    ObPsSqlKey ps_key;
    ps_key.db_id_ = db_id;
    ps_key.ps_sql_ = info_ctx.normalized_sql_;
    // add stmt item
    if (OB_FAIL(ps_cache->get_or_add_stmt_item(ps_key,
                                               is_contain_tmp_tbl,
                                               ps_stmt_item))) {
    } else if (OB_FAIL(ps_cache->get_or_add_stmt_info(info_ctx,
                                                      result,
                                                      schema_guard,
                                                      ps_stmt_item,
                                                      ref_stmt_info))) {
    } else if (OB_ISNULL(ps_stmt_item) || OB_ISNULL(ref_stmt_info)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      ref_stmt_info->set_literal_stmt_type(result.get_literal_stmt_type());
    }
    if (NULL != ref_stmt_info) {
      ref_stmt_info->set_is_sensitive_sql(info_ctx.is_sensitive_sql_);
    }
    //add session info
    if (OB_SUCC(ret)) {
      ObPsStmtId inner_stmt_id = ps_stmt_item->get_ps_stmt_id();
      ObPsStmtId client_stmt_id = OB_INVALID_ID;
      if (OB_FAIL(session.prepare_ps_stmt(inner_stmt_id,
                                          ref_stmt_info,
                                          client_stmt_id,
                                          duplicate_prepare,
                                          info_ctx.is_inner_sql_))) {
      } else {
        result.set_statement_id(client_stmt_id);
        result.set_stmt_type(info_ctx.stmt_type_);
        LOG_TRACE("add ps session info", K(ret), K(*ref_stmt_info), K(client_stmt_id),
                                        K(*ps_stmt_item), K(session.get_server_sid()));
      }
    }
    if (OB_FAIL(ret) || duplicate_prepare) { //dec ref count
      if (NULL != ps_stmt_item) {
        if (NULL != ref_stmt_info) {
          ObPsStmtId inner_stmt_id = ps_stmt_item->get_ps_stmt_id();
          ps_cache->deref_stmt_info(inner_stmt_id); // need to decide whether to remove
        }
        ps_stmt_item->dec_ref_count();
      }
    }
  }
  return ret;
}

int ObSql::check_contain_temporary_table(share::schema::ObSchemaGetterGuard &schema_guard,
                                         ObResultSet &result,
                                         bool &is_contain_tmp_tbl)
{
  int ret = OB_SUCCESS;
  is_contain_tmp_tbl = false;
  const ObTableSchema *table_schema = nullptr;
  for (int64_t i = 0; OB_SUCC(ret) && !is_contain_tmp_tbl && i < result.get_ref_objects().count(); i++) {
    table_schema = nullptr;
    ObSchemaObjVersion &obj_version = result.get_ref_objects().at(i);
    if (DEPENDENCY_TABLE != obj_version.object_type_) {
      // do nothing
    } else if (OB_FAIL(schema_guard.get_table_schema(
                                              obj_version.object_id_,
                                              table_schema))) {
    } else if (nullptr == table_schema) {
      ret = OB_ERR_UNEXPECTED;
    } else if (table_schema->is_tmp_table()) {
      is_contain_tmp_tbl = true;
      break;
    }
  }
  return ret;
}

int ObSql::do_real_prepare(const ObString &sql,
                           ObSqlCtx &context,
                           ObResultSet &result,
                           bool is_inner_sql)
{
  int ret = OB_SUCCESS;
  ParseResult parse_result;
  MEMSET(&parse_result, 0, SIZEOF(ParseResult));
  ObStmt *basic_stmt = NULL;
  stmt::StmtType stmt_type = stmt::T_NONE;
  int64_t param_cnt = 0;
  PsCacheInfoCtx info_ctx;
  ObIAllocator &allocator = result.get_mem_pool();
  ObSQLSessionInfo &session = result.get_session();
  ObExecContext &ectx = result.get_exec_context();
  ObParser parser(allocator, session.get_sql_mode(), session.get_charsets4parser());
  ParseMode parse_mode = (context.is_dynamic_sql_  || !is_inner_sql) ? DYNAMIC_SQL_MODE :
                         session.is_for_trigger_package() ? TRIGGER_MODE : STD_MODE;

  // normal ps sql also a dynamic sql, we adjust is_dynamic_sql_ for normal ps sql parser.
  context.is_dynamic_sql_ = !context.is_dynamic_sql_ ? !is_inner_sql : context.is_dynamic_sql_;

  bool is_from_pl = (NULL != context.secondary_namespace_ || result.is_simple_ps_protocol());
  ObPsPrepareStatusGuard ps_status_guard(session);
  ObPlanCacheCtx pc_ctx(sql, PC_PS_MODE, allocator, context, ectx);
  ParamStore param_store( (ObWrapperAllocator(&allocator)) );
  pc_ctx.set_is_inner_sql(is_inner_sql);

  if (OB_ISNULL(context.session_info_) || OB_ISNULL(context.schema_guard_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(parser.parse(sql,
                                  parse_result,
                                  parse_mode))) {
  } else if (ObSQLUtils::is_mysql_ps_not_support_stmt(parse_result)) {
    ret = OB_ER_UNSUPPORTED_PS;
  }  else if (parse_result.question_mark_ctx_.count_ > common::OB_MAX_PS_PARAM_COUNT) {
    ret = OB_ERR_PS_TOO_MANY_PARAM;
    LOG_USER_ERROR(OB_ERR_PS_TOO_MANY_PARAM);
  } else {
    ps_status_guard.is_varparams_sql_prepare(parse_result.question_mark_ctx_.count_ > 0 ? true : false);
  }
  context.is_sensitive_ |= parse_result.contain_sensitive_data_;

  OZ (ObResolverUtils::resolve_stmt_type(parse_result, stmt_type));

  if (OB_FAIL(ret)) {
  } else if (result.is_simple_ps_protocol() // simple_ps_protocol only do parse
             // for anonymous block, mock prepare only parses and reports bind metadata
             || (stmt::T_ANONYMOUS_BLOCK == stmt_type
                 && context.is_prepare_protocol_
                 && context.is_prepare_stage_
                 && context.is_mock_prepare_)) {
    param_cnt = parse_result.question_mark_ctx_.count_;
    info_ctx.normalized_sql_ = sql;
    if (stmt::T_ANONYMOUS_BLOCK == stmt_type
        && context.is_prepare_protocol_
        && context.is_prepare_stage_
        && context.is_mock_prepare_) {
      OZ (result.reserve_param_columns(param_cnt));
      for (int64_t i = 0; OB_SUCC(ret) && i < param_cnt; ++i) {
        ObField param_field;
        param_field.type_.set_type(ObIntType);
        param_field.cname_ = ObString::make_string("?");

        // Mock prepare does not fully resolve anonymous blocks, so out row info is unknown here.
        // Mark parameters as INOUT conservatively to avoid selecting an async driver that cannot return rows.

        param_field.inout_mode_ = ObRoutineParamInOut::SP_PARAM_INOUT;
        OZ (result.add_param_column(param_field), K(param_field), K(i), K(param_cnt));
      }
    }

  } else {
    if (context.is_dynamic_sql_) {
      parse_result.input_sql_ = parse_result.no_param_sql_;
      parse_result.input_sql_len_ = parse_result.no_param_sql_len_;
    }
    if (OB_FAIL(generate_stmt(parse_result, NULL, context, allocator, result, basic_stmt))) {
    } else if (ObStmt::is_dml_stmt(stmt_type)
              && !ObStmt::is_show_stmt(stmt_type)
              && !is_inner_sql
              && !is_from_pl) {
      bool is_transform_outline = false;
      if (OB_FAIL(ObSqlParameterization::parameterize_syntax_tree(allocator,
                                                                  is_transform_outline,
                                                                  pc_ctx,
                                                                  parse_result.result_tree_,
                                                                  param_store,
                                                                  session.get_charsets4parser()))) {
        LOG_INFO("parameterize syntax tree failed", K(ret));
        pc_ctx.ps_need_parameterized_ = false;
        ret = OB_SUCCESS;
      }
      if (OB_SUCC(ret)) {
        if (!pc_ctx.ps_need_parameterized_) {
          pc_ctx.fixed_param_idx_.reset();
          pc_ctx.fp_result_.raw_params_.reset();
        } else {
          info_ctx.no_param_sql_ = pc_ctx.sql_ctx_.plan_key_.constructed_sql_;
        }
      }
    } else if (!is_inner_sql && stmt::T_CALL_PROCEDURE == stmt_type) {
      if (OB_FAIL(ObSqlParameterization::parameterize_syntax_tree(allocator,
                                                                  false,
                                                                  pc_ctx,
                                                                  parse_result.result_tree_,
                                                                  param_store,
                                                                  session.get_charsets4parser()))) {
        LOG_INFO("parameterize call procedure syntax tree failed", K(ret));
        pc_ctx.ps_need_parameterized_ = false;
        pc_ctx.fixed_param_idx_.reset();
        pc_ctx.fp_result_.raw_params_.reset();
        ret = OB_SUCCESS;
      }
      if (OB_SUCC(ret)) {
        if (!pc_ctx.ps_need_parameterized_) {
          pc_ctx.fixed_param_idx_.reset();
          pc_ctx.fp_result_.raw_params_.reset();
        } else {
          info_ctx.no_param_sql_ = pc_ctx.sql_ctx_.plan_key_.constructed_sql_;
        }
      }
    }

    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(basic_stmt)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_ISNULL(basic_stmt->get_query_ctx())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (stmt::T_CALL_PROCEDURE == basic_stmt->get_stmt_type()
               && FALSE_IT(result.set_cmd(dynamic_cast<ObICmd*>(basic_stmt)))) {
    } else if (OB_FAIL(fill_result_set(result, &context, PC_PS_MODE, *basic_stmt))) {
    } else if (OB_ISNULL(result.get_param_fields())) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      param_cnt = result.get_param_fields()->count();
      stmt_type = basic_stmt->get_stmt_type();
      // If it is internal sql, such as pl internal sql, need to use the formatted text string,
      // Because need to replace the variables in sql of pl with standard ps text for hard parsing
      // and the ps text of external requests cannot be formatted, because checksum verification is required when parsing the ps execute package,
      // Need to ensure the prepare text is consistent with what the client sends
      if (is_inner_sql) {
        // pl
        info_ctx.normalized_sql_ = basic_stmt->get_query_ctx()->get_sql_stmt();
      } else {
        info_ctx.normalized_sql_ = sql;
      }
    }

    LOG_INFO("generate new stmt", K(ret), K(param_cnt), K(stmt_type), K(info_ctx.no_param_sql_),
             K(info_ctx.normalized_sql_),
             "sql", context.is_sensitive_ ? ObString(OB_MASKED_STR) : sql);
  }
  if (OB_SUCC(ret)) {
    info_ctx.param_cnt_ = param_cnt;
    info_ctx.stmt_type_ = stmt_type;
    info_ctx.is_inner_sql_ = is_inner_sql;
    info_ctx.is_sensitive_sql_ = context.is_sensitive_;
    info_ctx.raw_params_ = &pc_ctx.fp_result_.raw_params_;
    info_ctx.fixed_param_idx_ = &pc_ctx.fixed_param_idx_;
    info_ctx.raw_sql_.assign_ptr(sql.ptr(), sql.length());
    if (OB_FAIL(do_add_ps_cache(info_ctx, *context.schema_guard_, result))) {
    }
  }
  //if the error code is ob_timeout, we add more error info msg for dml query.
  if (OB_UNLIKELY(OB_TIMEOUT == ret && session.is_user_session())) {
    LOG_USER_ERROR(OB_TIMEOUT, THIS_WORKER.get_timeout_ts() - session.get_query_start_time());
  }
  LOG_INFO("add ps cache", K(info_ctx.normalized_sql_), K(param_cnt), K(ret));
  return ret;
}


ObSql::TimeoutGuard::TimeoutGuard(ObSQLSessionInfo &session)
  : session_(session)
{
  int ret = OB_SUCCESS;
  worker_timeout_ = THIS_WORKER.get_timeout_ts();
}

ObSql::TimeoutGuard::~TimeoutGuard()
{
  int ret = OB_SUCCESS;
  if (THIS_WORKER.get_timeout_ts() != worker_timeout_) {
    THIS_WORKER.set_timeout_ts(worker_timeout_);
  }
}

int ObSql::set_timeout_for_pl(ObSQLSessionInfo &session_info, int64_t &abs_timeout_us)
{
  int ret = OB_SUCCESS;
  int64_t query_timeout;
  if (THIS_WORKER.is_timeout()) {
    ret = OB_TIMEOUT;
  } else if (OB_FAIL(session_info.get_query_timeout(query_timeout))) {
    // do nothing
  } else {
    OX (abs_timeout_us = session_info.get_query_start_time() > 0
                         ? session_info.get_query_start_time() + query_timeout
                           : ObTimeUtility::current_time() + query_timeout);
    if (THIS_WORKER.get_timeout_ts() > abs_timeout_us) {
      OX (THIS_WORKER.set_timeout_ts(abs_timeout_us));
    }
  }
  return ret;
}

/*
 * sql: sql statement in pl
 * PLPrepareCtx: relevant information used for prepare
 * PLPrepareResult: output result after prepare
 */

int ObSql::prepare_pl_sql(const ObString &sql,
                          ObSPIService::PLPrepareCtx &pl_prepare_ctx,
                          ObSPIService::PLPrepareResult &pl_prepare_result,
                          ParamStore *params)
{
  int ret = OB_SUCCESS;
  ObString cur_query;
  ObString trimed_stmt = const_cast<ObString &>(sql).trim();
  ObSqlCtx &context = pl_prepare_result.sql_ctx_;
  int64_t param_cnt = 0;
  ObString normalized_sql;
  ParseResult parse_result;
  MEMSET(&parse_result, 0, SIZEOF(ParseResult));
  ObStmt *basic_stmt = NULL;
  int64_t cur_timeout_us = 0;
  stmt::StmtType stmt_type = stmt::T_NONE;
  ObSchemaGetterGuard &schema_guard = pl_prepare_result.schema_guard_;
  ObSQLSessionInfo &sess = pl_prepare_ctx.sess_info_;
  TimeoutGuard timeout_guard(sess);
  int64_t old_query_start_time = sess.get_query_start_time();
  sess.set_query_start_time(ObTimeUtility::current_time());
  OZ (pl_prepare_result.init(sess, get_plan_cache_access_service()));
  CK (OB_NOT_NULL(pl_prepare_result.get_allocator()));
  CK (OB_NOT_NULL(pl_prepare_result.result_set_));
  if (OB_SUCC(ret)) {
    ObIAllocator &allocator = *pl_prepare_result.get_allocator();
    ObParser parser(allocator, sess.get_sql_mode(), sess.get_charsets4parser());
    bool is_for_trigger = false;
    if (NULL != pl_prepare_ctx.secondary_ns_) {
      is_for_trigger = ObTriggerInfo::is_trigger_package_id(pl_prepare_ctx.secondary_ns_->get_package_id());
    }
    ParseMode parse_mode = pl_prepare_ctx.is_dynamic_sql_ ? DYNAMIC_SQL_MODE :
                          is_for_trigger ? TRIGGER_MODE : STD_MODE;

    context.is_dynamic_sql_ = pl_prepare_ctx.is_dynamic_sql_;
    context.is_cursor_ = pl_prepare_ctx.is_cursor_;
    context.secondary_namespace_ = pl_prepare_ctx.secondary_ns_;
    context.session_info_ = &sess;
    context.disable_privilege_check_ = PRIV_CHECK_FLAG_DISABLE;
    context.exec_type_ = PLSql;
    context.is_prepare_protocol_ = true;
    context.is_prepare_stage_ = true;

    if (OB_FAIL(ob_write_string(allocator, sess.get_current_query_string(), cur_query))) {
    }

    if (OB_SUCC(ret)) {
      WITH_CONTEXT(pl_prepare_result.mem_context_) {
          ObResultSet &result = *pl_prepare_result.result_set_;
          LinkExecCtxGuard link_guard(result.get_session(), result.get_exec_context());
          ObPsPrepareStatusGuard ps_status_guard(sess);
          if (trimed_stmt.empty()) {
            ret = OB_ERR_EMPTY_QUERY;
          } else if (OB_FAIL(set_timeout_for_pl(sess, cur_timeout_us))) {
          } else if (OB_FAIL(GCTX.schema_service_->get_runtime_schema_guard(
                                  schema_guard))) {
          } else if (FALSE_IT(context.schema_guard_ = &schema_guard)) {
          } else if (OB_FAIL(init_result_set(context, result))) {
          } else if (OB_ISNULL(result.get_exec_context().get_physical_plan_ctx())) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_NOT_NULL(params) &&
                     OB_FAIL(construct_param_store(*params,
                                                   result.get_exec_context().get_physical_plan_ctx()->get_param_store_for_update()))) {
          } else if (OB_FAIL(sess.store_query_string(sql))) {
          } else if (OB_FAIL(parser.parse(sql, parse_result, parse_mode,
                                          false, false, true,
                                          pl_prepare_ctx.is_parser_dynamic_sql_))) {
          } else if (ObSQLUtils::is_mysql_ps_not_support_stmt(parse_result)) {
            ret = OB_ER_UNSUPPORTED_PS;
          } else if (NULL == pl_prepare_ctx.secondary_ns_ && !pl_prepare_ctx.is_dynamic_sql_) {
            result.set_simple_ps_protocol();
          }
          context.is_sensitive_ |= parse_result.contain_sensitive_data_;

          if (OB_SUCC(ret) && !pl_prepare_ctx.is_parser_dynamic_sql_) {
            ps_status_guard.is_varparams_sql_prepare(parse_result.question_mark_ctx_.count_ > 0 ? true : false);
          }
          if (OB_FAIL(ret)) {
            // do nothing
          } else if (OB_FAIL(ObResolverUtils::resolve_stmt_type(parse_result, stmt_type))) {
          } else if (FALSE_IT(result.set_stmt_type(stmt_type))) {
          } else if (result.is_simple_ps_protocol()
                    || (stmt::T_ANONYMOUS_BLOCK == stmt_type && context.is_prepare_protocol_
                        && context.is_prepare_stage_ && context.is_mock_prepare_)) {
            if (parse_result.is_dynamic_sql_) {
              context.is_dynamic_sql_ = true;
            }
            bool for_update = false;
            if (stmt::T_SELECT == stmt_type &&
                PARSE_SELECT_MAX_IDX == parse_result.result_tree_->children_[0]->num_child_ &&
                NULL != parse_result.result_tree_->children_[0]->children_[PARSE_SELECT_FOR_UPD]) {
              for_update = true;
            }
            result.get_external_retrieve_info().is_select_for_update_ = for_update;
            param_cnt = parse_result.question_mark_ctx_.count_;
            normalized_sql = context.is_dynamic_sql_ && parse_result.no_param_sql_len_ > 0
              ? ObString(parse_result.no_param_sql_len_, parse_result.no_param_sql_) : sql;
            if (stmt::T_ANONYMOUS_BLOCK == stmt_type && context.is_prepare_protocol_
                && context.is_prepare_stage_ && context.is_mock_prepare_) {
              OZ (result.reserve_param_columns(param_cnt));
              for (int64_t i = 0; OB_SUCC(ret) && i < param_cnt; ++i) {
                ObField param_field;
                param_field.type_.set_type(ObIntType);
                param_field.cname_ = ObString::make_string("?");
                OZ (result.add_param_column(param_field), K(param_field), K(i), K(param_cnt));
              }
            }
          } else {
            if (parse_result.is_dynamic_sql_) {
              context.is_dynamic_sql_ = true;
            }
            if (context.is_dynamic_sql_) {
              parse_result.input_sql_ = parse_result.no_param_sql_;
              parse_result.input_sql_len_ = parse_result.no_param_sql_len_;
            }
            pl_prepare_result.question_mark_cnt_ = parse_result.question_mark_ctx_.count_;
            if (OB_FAIL(generate_stmt(parse_result, NULL, context, allocator, result, basic_stmt))) {
            } else if (OB_ISNULL(basic_stmt)) {
              ret = OB_ERR_UNEXPECTED;
            } else if (OB_ISNULL(basic_stmt->get_query_ctx())) {
              ret = OB_ERR_UNEXPECTED;
            } else if (stmt::T_CALL_PROCEDURE == basic_stmt->get_stmt_type()
                      && FALSE_IT(result.set_cmd(dynamic_cast<ObICmd*>(basic_stmt)))) {
            } else if (OB_FAIL(fill_result_set(result, &context, PC_PL_MODE, *basic_stmt))) {
            } else if (OB_ISNULL(result.get_param_fields())) {
              ret = OB_INVALID_ARGUMENT;
            } else {
              normalized_sql = basic_stmt->get_query_ctx()->get_sql_stmt();
            }
          }

          if (OB_FAIL(ret)) {
            // do nothing
          } else if (OB_FAIL(ob_write_string(allocator, normalized_sql, result.get_stmt_ps_sql(), true))) {
          }
          int tmp_ret = OB_SUCCESS;
          if ((tmp_ret = sess.store_query_string(cur_query)) != OB_SUCCESS) {
            LOG_WARN("failed to store query string", K(ret), K(tmp_ret));
            ret = OB_SUCCESS == ret ?  tmp_ret : ret;
          }
      }
    }
  }
  sess.set_query_start_time(old_query_start_time);
  return ret;
}

int ObSql::handle_sql_execute(const ObString &sql,
                              ObSqlCtx &context,
                              ObResultSet &result,
                              ParamStore &org_params,
                              PlanCacheMode mode)
{
  int ret = OB_SUCCESS;
  int get_plan_err = OB_SUCCESS;
  bool use_plan_cache = false;
  ObIAllocator &allocator = result.get_mem_pool();
  ObSQLSessionInfo *session = context.session_info_;
  ObExecContext &ectx = result.get_exec_context();
  ObPhysicalPlanCtx *pctx = ectx.get_physical_plan_ctx();
  ParamStore params( (ObWrapperAllocator(allocator)) );
  ParamStore *ab_params = NULL;

  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(session) || OB_ISNULL(plan_cache_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if ((mode == PC_PS_MODE || mode == PC_PL_MODE) && OB_ISNULL(pctx)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    use_plan_cache = session->get_local_ob_enable_plan_cache();
  }

  ObPlanCacheCtx pc_ctx(sql, mode, allocator, context, ectx);

  if (OB_FAIL(ret)) {
    // do nothing
  } else if (mode == PC_PL_MODE) {
    if (OB_FAIL(reconstruct_pl_params_store(allocator, context, org_params, params, ab_params))) {
    } else if (context.is_batch_params_execute() && OB_ISNULL(ab_params)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(construct_param_store(params, pctx->get_param_store_for_update()))) {
    } else if (OB_FAIL(construct_parameterized_params(params, pc_ctx))) {
    } else {
      pc_ctx.normal_parse_const_cnt_ = params.count();
      pc_ctx.set_is_parameterized_execute();
      pc_ctx.ab_params_ = ab_params;
    }
  } else if (mode == PC_PS_MODE) {
    // TODO, not support ps now.
    ret = OB_ERR_UNEXPECTED;
  }

  if (OB_SUCC(ret)) {
    if (!use_plan_cache) {
      // do nothing
    } else if (OB_FAIL(pc_get_plan_and_fill_result(pc_ctx, result,
                         get_plan_err, ectx.get_need_disconnect_for_update()))) {
    }
  }

  if (OB_SUCC(ret)) {
    if (!result.get_is_from_plan_cache()) {
      if (mode == PC_PS_MODE || mode == PC_PL_MODE) {
        pctx->get_param_store_for_update().reset();
      }
      // Use SMART_CALL_LARGE for complex SQL plan generation when plan cache misses
      // This prevents frequent stack extending during physical plan generation
      if (OB_FAIL(SMART_CALL_LARGE(handle_physical_plan(sql, context, result, pc_ctx, get_plan_err)))) {
      }
    }
  }

  if (OB_SUCC(ret) && !context.is_text_ps_mode_) {
    if (OB_FAIL(after_get_plan(pc_ctx, *session, result.get_physical_plan(),
                result.get_is_from_plan_cache()))) {
    }
  }

  return ret;
}

/*!
 * sql: The SQL statement to be executed
 * params: The parameter list for the current SQL statement
 * res: Direct result set
 */
// TODO remove is_prepare_protocol and is_dynamic_sql
int ObSql::execute_pl_sql(const ObString &sql,
                          ObSQLSessionInfo &session,
                          ParamStore &params,
                          ObResultSet &result,
                          ObSqlCtx &context,
                          bool is_prepare_protocol,
                          bool is_dynamic_sql)
{
  int ret = OB_SUCCESS;
  int get_plan_err = OB_SUCCESS;
  TimeoutGuard timeout_guard(session);
  LinkExecCtxGuard link_guard(result.get_session(), result.get_exec_context());
  int64_t cur_timeout_us = 0;
  context.session_info_ = &session;
  ObIAllocator &allocator = result.get_mem_pool();
  ObExecContext &ectx = result.get_exec_context();
  ObPhysicalPlanCtx *pctx = NULL;

  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(result.init())) {
  } else if (OB_FAIL(init_result_set(context, result))) {
  } else {
    context.cur_sql_ = sql;
    context.is_from_pl_ = true;
    context.is_dynamic_sql_ = is_dynamic_sql;
    context.is_prepare_protocol_ = is_prepare_protocol;
    context.plan_key_.db_id_ = session.get_database_id();
    context.disable_privilege_check_ = PRIV_CHECK_FLAG_IN_PL;
    pctx = ectx.get_physical_plan_ctx();
    int64_t local_database_schema_version = -1;
    if (OB_ISNULL(context.schema_guard_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(context.schema_guard_->get_schema_version(local_database_schema_version))) {
    } else {
      result.get_exec_context().get_sql_exec_ctx().set_query_begin_schema_version(local_database_schema_version);
    }
  }
  if (OB_SUCC(ret) && is_prepare_protocol && !is_dynamic_sql) {
    result.set_simple_ps_protocol();
  }

  LOG_TRACE("arrive handle pl execute", K(ret),
            "sql", context.is_sensitive_ ? ObString(OB_MASKED_STR) : sql,
            K(is_prepare_protocol), K(is_dynamic_sql), K(lbt()));
  int64_t start_time = ObTimeUtility::current_time();
  if (OB_FAIL(ret)) {
  } else if (OB_ISNULL(pctx)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(set_timeout_for_pl(session, cur_timeout_us))) {
  } else if (OB_FAIL(session.store_query_string(sql))) {
  } else if (OB_FAIL(handle_sql_execute(sql, context, result, params, is_prepare_protocol ? PC_PL_MODE : PC_TEXT_MODE))) {
  }
  int64_t end_time = ObTimeUtility::current_time();
  if (OB_FAIL(ret) && OB_SUCCESS == result.get_errcode()) {
    result.set_errcode(ret);
  }
  //todo:@hr351303 the logic below will be moved to the spi layer later
  if (OB_SUCC(ret)) {
    if (OB_ISNULL(context.schema_guard_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(session.update_query_sensitive_system_variable(*(context.schema_guard_)))) {
    } else if (OB_FAIL(result.open())) {
    } else {
      // do nothing
    }
  }

  if (OB_SUCC(ret) && session.get_in_transaction()) {
    if (ObStmt::is_dml_write_stmt(result.get_stmt_type()) ||
        ObStmt::is_savepoint_stmt(result.get_stmt_type()) ||
        (ObStmt::is_select_stmt(result.get_stmt_type()) && OB_NOT_NULL(result.get_physical_plan()) && result.get_physical_plan()->has_for_update())) {
      session.set_has_exec_inner_dml(true);
    }
  }
  return ret;
}

int ObSql::handle_ps_prepare(const ObString &stmt,
                             ObSqlCtx &context,
                             ObResultSet &result,
                             bool is_inner_sql)
{
// open_cursors is 0 to indicate a special state, no limit is set
#define NEED_CHECK_SESS_MAX_PS_HANDLE_LIMIT(v) (0 == v ? false : true)
  int ret = OB_SUCCESS;
  ObString cur_query;
  // trimmed_stmt is only used for query empty check, prepare statement needs to use the original statement, avoid checksum inconsistency
  ObString trimed_stmt = const_cast<ObString &>(stmt).trim();
  if (trimed_stmt.empty()) {
    ret = OB_ERR_EMPTY_QUERY;
  } else if (OB_FAIL(init_result_set(context, result))) {
  }

  if (OB_SUCC(ret)) {
    ObSQLSessionInfo &session = result.get_session();
    ObPsCache *ps_cache = ps_cache_;
    ObExecContext &ectx = result.get_exec_context();
    ObIAllocator &allocator = result.get_mem_pool();
    ObPhysicalPlanCtx *pctx = ectx.get_physical_plan_ctx();
    ObSchemaGetterGuard *schema_guard = context.schema_guard_;
    ectx.set_is_ps_prepare_stage(true);

#ifndef NDEBUG
    LOG_INFO("Begin to handle prepare statement", "sess_id", session.get_server_sid(), K(stmt));
#endif

    if (OB_ISNULL(ps_cache) || OB_ISNULL(pctx) || OB_ISNULL(schema_guard)) {
      ret = OB_INVALID_ARGUMENT;
      LOG_ERROR("physical plan context or ps plan cache is NULL or schema_guard is null",
                K(ret), K(pctx), K(ps_cache));
    } else if (OB_FAIL(ob_write_string(allocator, session.get_current_query_string(), cur_query))) {
    } else if (OB_FAIL(session.store_query_string(stmt))) {
    } else {
      bool need_do_real_prepare = false;
      uint64_t db_id = OB_INVALID_ID;
      (void)session.get_database_id(db_id);
      ObPsSqlKey ps_key;
      ps_key.db_id_ = db_id;
      ps_key.ps_sql_ = stmt;
      ObPsStmtId inner_stmt_id = OB_INVALID_STMT_ID;
      ObPsStmtId client_stmt_id = OB_INVALID_STMT_ID;
      ObPsStmtInfo *stmt_info = NULL;
      ObPsStmtItem *stmt_item = NULL;
      bool duplicate_prepare = false;
      bool is_expired = false;
      int64_t open_cursors_limit = 0;
      int64_t cur_ps_handle_size = session.get_ps_session_info_size();
      if (FALSE_IT(open_cursors_limit = config::open_cursors())) {
      } else if (!is_inner_sql
                && NEED_CHECK_SESS_MAX_PS_HANDLE_LIMIT(open_cursors_limit)
                && cur_ps_handle_size >= open_cursors_limit) {
        ret = OB_ERR_OPEN_CURSORS_EXCEEDED;
      } else if (NULL != context.secondary_namespace_ || result.is_simple_ps_protocol()) {
        // pl initiated sql parsing, since each time it needs to calculate dependent objects and other extra parameters, therefore do_real_prepare needs to be done
        need_do_real_prepare = true;
        if (REACH_TIME_INTERVAL(1000000)) {
          LOG_INFO("need do real prepare",
                   K(db_id), K(stmt), K(need_do_real_prepare), K(context.secondary_namespace_),
                   K(result.is_simple_ps_protocol()));
        }
      } else if (OB_FAIL(ps_cache->ref_stmt_item(ps_key, stmt_item))) {
        if (OB_HASH_NOT_EXIST == ret) {
          ret = OB_SUCCESS;
          need_do_real_prepare = true;
          if (REACH_TIME_INTERVAL(1000000)) {
            LOG_INFO("stmt id not exist", K(db_id), K(stmt), K(need_do_real_prepare));
          }
        } else {
        }
      } else if (OB_ISNULL(stmt_item)
                 || OB_INVALID_STMT_ID == (inner_stmt_id = stmt_item->get_ps_stmt_id())) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ps_cache->ref_stmt_info(inner_stmt_id, stmt_info))) {
        //inc stmt_info ref for session
        if (OB_HASH_NOT_EXIST == ret) {
          need_do_real_prepare = true;
          if (REACH_TIME_INTERVAL(1000000)) {
            LOG_INFO("stmt info not exist", K(db_id), K(stmt), K(inner_stmt_id), K(ret));
          }
          ret = OB_SUCCESS;
        } else {
        }
      } else if (OB_ISNULL(stmt_info)) {
        ret = OB_INVALID_ARGUMENT;
        //check stmt_info whether expired, if expired, do nothing
      } else if (OB_FAIL(ps_cache->check_schema_version(*context.schema_guard_,
                                                        *stmt_info,
                                                        is_expired))) {
      } else if (is_expired) {
        stmt_info->set_is_expired();
        if (OB_FAIL(ps_cache->erase_stmt_item(inner_stmt_id, ps_key))) {
        }
        need_do_real_prepare = true;
      } else if (OB_FAIL(session.prepare_ps_stmt(inner_stmt_id,
                                                stmt_info,
                                                client_stmt_id,
                                                duplicate_prepare,
                                                is_inner_sql))) {
      } else if (OB_FAIL(fill_result_set(client_stmt_id, *stmt_info, result))) {
        // prepare ps stmt succeeded, failure here requires close
        IGNORE_RETURN session.close_ps_stmt(*ps_cache, client_stmt_id);
      }
      if (OB_FAIL(ret)
          || need_do_real_prepare
          || duplicate_prepare) {
        if (NULL != stmt_item) {
          stmt_item->dec_ref_count();
        }
        if (NULL != stmt_info) {
          ps_cache->deref_stmt_info(inner_stmt_id); // need to decide whether to remove
        }
      }
      if (OB_SUCC(ret) && need_do_real_prepare) {
        if (OB_FAIL(do_real_prepare(stmt, context, result, is_inner_sql))) {
        }
      } else if (OB_SUCC(ret) && NULL != stmt_info) {
        context.is_sensitive_ = stmt_info->get_is_sensitive_sql();
      }
      if (OB_SUCC(ret)) {
        if (false == need_do_real_prepare) {
          ps_cache->inc_access_and_hit_count();
        } else {
          // Without hitting ps cache, only increase access count
          // The judgment logic here will cause pl to only increase the access count, not the hit_count, every prepare
          // So the ps cache hit rate seen from the ps related virtual tables will be relatively low
          ps_cache->inc_access_count();
        }
      }
    }
    OZ (session.store_query_string(cur_query));
  }
  return ret;
}

int ObSql::add_param_to_param_store(const ObObjParam &param,
                                    ParamStore &param_store)
{
  int ret = OB_SUCCESS;
  if (param.is_numeric_type()) {
    const_cast<ObObjParam &>(param).set_param_meta();
  }
  if (OB_FAIL(param_store.push_back(param))) {
  }
  return ret;
}

int ObSql::construct_param_store_from_parameterized_params(const ObPlanCacheCtx &phy_ctx,
                                               ParamStore &param_store)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(param_store.reserve(phy_ctx.fp_result_.parameterized_params_.count()))) {
  }
  for (int i = 0; OB_SUCC(ret) && i < phy_ctx.fp_result_.parameterized_params_.count(); ++i) {
    const common::ObObjParam *param = phy_ctx.fp_result_.parameterized_params_.at(i);
    if (OB_FAIL(add_param_to_param_store(*param, param_store))) {
    }
  }
  return ret;
}

bool ObSql::is_exist_in_fixed_param_idx(const int64_t idx,
                                        const ObIArray<int64_t> &fixed_param_idx)
{
  bool bool_ret = false;
  for (int i = 0; i < fixed_param_idx.count(); ++i) {
    if (idx == fixed_param_idx.at(i)) {
      bool_ret = true;
      break;
    }
  }
  return bool_ret;
}

int ObSql::construct_ps_param_store(const ParamStore &params,
                                    const ParamStore &fixed_params,
                                    const ObIArray<int64_t> &fixed_params_idx,
                                    ParamStore &param_store)
{
  int ret = OB_SUCCESS;
  int64_t param_idx = 0;
  int64_t fixed_param_idx = 0;
  int64_t param_count = params.count() + fixed_params.count();
  if (fixed_params.count() != fixed_params_idx.count()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("the number of fixed param does not match", K(fixed_params), K(fixed_params_idx));
  } else if (OB_FAIL(param_store.reserve(param_count))) {
  }
  for (int i = 0; OB_SUCC(ret) && i < param_count; ++i) {
    if (is_exist_in_fixed_param_idx(i, fixed_params_idx)) {
      if (fixed_param_idx >= fixed_params.count()) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Invalid index", K(fixed_param_idx), K(fixed_params));
      } else if (OB_FAIL(add_param_to_param_store(fixed_params.at(fixed_param_idx),
                                                  param_store))) {
      } else {
        LOG_TRACE("ps param is", K(fixed_params.at(fixed_param_idx)), K(i));
      }
      ++fixed_param_idx;
    } else {
      if (param_idx >= params.count()) {
        ret = OB_ERR_UNEXPECTED;
        LOG_WARN("Invalid index", K(param_idx), K(params));
      } else if (OB_FAIL(add_param_to_param_store(params.at(param_idx), param_store))) {
      } else {
        LOG_TRACE("ps param is", K(params.at(param_idx)), K(i));
      }
      ++param_idx;
    }
  }
  return ret;
}

int ObSql::construct_param_store(const ParamStore &params,
                                 ParamStore &param_store)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(param_store.reserve(params.count()))) {
  }
  for (int i = 0; OB_SUCC(ret) && i < params.count(); ++i) {
    if (OB_FAIL(add_param_to_param_store(params.at(i), param_store))) {
    }
    LOG_TRACE("ps param is", K(params.at(i)), K(i));
  }
  return ret;
}

int ObSql::construct_parameterized_params(const ParamStore &params,
                                          ObPlanCacheCtx &phy_ctx)
{
  int ret = OB_SUCCESS;
  phy_ctx.fp_result_.parameterized_params_.reset();
  phy_ctx.fp_result_.parameterized_params_.set_allocator(&phy_ctx.allocator_);
  phy_ctx.fp_result_.parameterized_params_.set_capacity(params.count());
  for (int i = 0; OB_SUCC(ret) && i < params.count(); ++i) {
    if (OB_FAIL(phy_ctx.fp_result_.parameterized_params_.push_back(&params.at(i)))) {
    }
  }
  return ret;
}

int ObSql::clac_fixed_param_store(const stmt::StmtType stmt_type,
                                  const ObIArray<int64_t> &raw_params_idx,
                                  const ObIArray<ObPCParam *> &raw_params,
                                  ObIAllocator &allocator,
                                  ObSQLSessionInfo &session,
                                  ParamStore &fixed_param_store)
{
  int ret = OB_SUCCESS;
  ObCollationType collation_connection = static_cast<ObCollationType>(
                                         session.get_local_collation_connection());
  ObString literal_prefix;
  ObObjParam value;
  const bool is_paramlize = false;
  int64_t server_collation = CS_TYPE_INVALID;
  bool enable_decimal_int = false;
  bool enable_mysql_compatible_dates = false;
  if (raw_params.empty()) {
    // do nothing
  } else if (raw_params_idx.count() != raw_params.count()) {
    ret = OB_ERR_UNEXPECTED;
    LOG_WARN("the number of fixed param does not match",
    K(raw_params_idx.count()), K(raw_params.count()));
  } else if (OB_FAIL(fixed_param_store.reserve(raw_params_idx.count()))) {
  } else if (OB_FAIL(ObSQLUtils::check_enable_decimalint(&session, enable_decimal_int))) {
  } else if (OB_FAIL(ObSQLUtils::check_enable_mysql_compatible_dates(&session, false /*is_ddl*/,
                       enable_mysql_compatible_dates))) {
  }
  for (int i = 0; OB_SUCC(ret) && i < raw_params.count(); ++i) {
    value.reset();
    ParseNode *raw_param = NULL;;
    if (OB_ISNULL(raw_params.at(i))) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_ISNULL(raw_param = raw_params.at(i)->node_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(ObResolverUtils::resolve_const(raw_param,
                                                      stmt_type,
                                                      allocator,
                                                      collation_connection,
                                                      session.get_nls_collation_nation(),
                                                      session.get_timezone_info(),
                                                      value,
                                                      is_paramlize,
                                                      literal_prefix,
                                                      session.get_actual_length_semantics(),
                                                      static_cast<ObCollationType>(server_collation),
                                                      NULL, session.get_sql_mode(),
                                                      enable_decimal_int,
                                                      share::COMPAT_MYSQL57,
                                                      enable_mysql_compatible_dates,
                                                      session.get_min_const_integer_precision(),
                                                      stmt::T_ANONYMOUS_BLOCK == stmt_type))) {
    } else if (OB_FAIL(add_param_to_param_store(value, fixed_param_store))) {
    } else {
    }
  }
  return ret;
}

int ObSql::init_execute_params_for_ab(ObIAllocator &allocator,
                                      const ParamStore &params_store,
                                      ParamStore *&first_group_params)
{
  int ret = OB_SUCCESS;
  if (OB_NOT_NULL(first_group_params)) {
    // do nothing
  } else if (OB_ISNULL(first_group_params = static_cast<ParamStore *>(allocator.alloc(sizeof(ParamStore))))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (FALSE_IT(first_group_params = new(first_group_params)ParamStore(ObWrapperAllocator(allocator)))) {
  }
  if (OB_FAIL(ret)) {
    // do nothing
  } else if (OB_FAIL(ObPlanCacheValue::get_one_group_params(0, params_store, *first_group_params))) {
  } else {
    for (int64_t i = 0; OB_SUCC(ret) && i < first_group_params->count(); i++) {
      ObObjParam &obj_param = first_group_params->at(i);
      const ObObjParam &src_param = params_store.at(i);
      const ObSqlArrayObj *array_obj = NULL;
      if (OB_UNLIKELY(!src_param.is_ext_sql_array())) {
        // do not need set accuracy
      } else if (OB_ISNULL(array_obj = reinterpret_cast<const ObSqlArrayObj*>(src_param.get_ext()))) {
        ret = OB_ERR_UNEXPECTED;
      } else {
        obj_param.set_accuracy(array_obj->element_.get_accuracy());
      }
      obj_param.get_param_flag().is_batch_parameter_ = true;
    }
  }
  return ret;
}

int ObSql::reconstruct_pl_params_store(ObIAllocator &allocator,
                                       ObSqlCtx &context,
                                       const ParamStore &origin_params,
                                       ParamStore &pl_params,
                                       ParamStore *&pl_ab_params)
{
  int ret = OB_SUCCESS;
  if (context.is_batch_params_execute()) {
    ParamStore *first_group_params = &pl_params;
    if (OB_FAIL(init_execute_params_for_ab(allocator, origin_params, first_group_params))) {
    } else if (OB_ISNULL(pl_ab_params = static_cast<ParamStore *>(allocator.alloc(sizeof(ParamStore))))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (FALSE_IT(pl_ab_params = new(pl_ab_params)ParamStore(ObWrapperAllocator(allocator)))) {
      // do nothing
    } else if (OB_FAIL(construct_param_store(origin_params,
                                             *pl_ab_params))) {
    }
  } else {
    if (OB_FAIL(construct_param_store(origin_params, pl_params))) {
    }
  }
  return ret;
}

int ObSql::reconstruct_ps_params_store(ObIAllocator &allocator,
                                       ObSqlCtx &context,
                                       const ParamStore &origin_params,
                                       const ParamStore &fixed_params,
                                       ObPsStmtInfo *ps_info,
                                       ParamStore &ps_params,
                                       ParamStore *&ps_ab_params)
{
  int ret = OB_SUCCESS;
  ParamStore *first_group_params = NULL;
  if (context.is_batch_params_execute()) {
    if (OB_FAIL(init_execute_params_for_ab(allocator, origin_params, first_group_params))) {
    } else if (OB_FAIL(construct_ps_param_store(*first_group_params,
                                                fixed_params,
                                                ps_info->get_raw_params_idx(),
                                                ps_params))) {
    } else if (OB_ISNULL(ps_ab_params = static_cast<ParamStore *>(allocator.alloc(sizeof(ParamStore))))) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
    } else if (FALSE_IT(ps_ab_params = new(ps_ab_params)ParamStore(ObWrapperAllocator(allocator)))) {
      // do nothing
    } else if (OB_FAIL(construct_ps_param_store(origin_params,
                                                fixed_params,
                                                ps_info->get_raw_params_idx(),
                                                *ps_ab_params))) {
    }
  } else if (OB_FAIL(construct_ps_param_store(origin_params,
                                              fixed_params,
                                              ps_info->get_raw_params_idx(),
                                              ps_params))) {
  }
  return ret;
}

int ObSql::check_read_only_privilege(ParseResult &parse_result,
                                     ObExecContext &exec_ctx,
                                     ObSchemaGetterGuard &schema_guard,
                                     ObSqlTraits &sql_traits)
{
  int ret = OB_SUCCESS;
  bool read_only = false;
  ObPhysicalPlanCtx *pctx = exec_ctx.get_physical_plan_ctx();
  ObSQLSessionInfo *session = exec_ctx.get_my_session();
  sql_traits.is_readonly_stmt_ = ObSQLUtils::is_readonly_stmt(parse_result);
  sql_traits.is_cause_implicit_commit_
      = ObSQLUtils::cause_implicit_commit(parse_result);
  sql_traits.is_commit_stmt_ = ObSQLUtils::is_commit_stmt(parse_result);
  sql_traits.stmt_type_ = ObSQLUtils::get_sql_item_type(parse_result);
  const bool is_write_enabled = share::server_is_write_enabled();
  const bool is_standby_control_stmt = !is_write_enabled
      && ObSQLUtils::is_allowed_on_standby(sql_traits.stmt_type_);
  if (OB_ISNULL(pctx) || OB_ISNULL(session)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!(session->is_inner() && !session->is_user_session())
             && !is_write_enabled
             && !sql_traits.is_readonly_stmt_
             && !ObSQLUtils::is_allowed_on_standby(sql_traits.stmt_type_)) {
    ret = OB_STANDBY_DATABASE_READ_ONLY;
    LOG_WARN("standby server is read only", KR(ret), K(sql_traits));
  } else if (!is_standby_control_stmt
             && OB_FAIL(schema_guard.get_runtime_read_only(read_only))) {
  } else if (!is_standby_control_stmt
             && OB_FAIL(session->check_read_only_privilege(read_only,
                                                           sql_traits))) {
    LOG_WARN("failed to check read_only privilege", K(ret));
    if (ObSQLUtils::is_end_trans_stmt(parse_result)) {
      int et_ret = OB_SUCCESS;
      exec_ctx.set_need_disconnect(false);
      //FIXME qianfu NG_TRACE_EXT(set_need_disconnect, OB_ID(need_disconnect), false);
      LOG_WARN("is commit or rollback stmt, but fail to check read_only privilege, "
              "rollback", K(ret));
      int64_t plan_timeout = 0;
      if (OB_SUCCESS != (et_ret = session->get_query_timeout(plan_timeout))) {
      } else {
        pctx->set_timeout_timestamp(session->get_query_start_time() + plan_timeout);
        // explicitly rollback the transaction, if it fails, the connection will be disconnected
        if (OB_SUCCESS != (et_ret = ObSqlTransControl::explicit_end_trans(
                    exec_ctx, true))) {
        }
      }
    }
  }
  return ret;
}

int ObSql::handle_ps_execute(const ObPsStmtId client_stmt_id,
                             const stmt::StmtType stmt_type,
                             const ParamStore &params,
                             ObSqlCtx &context,
                             ObResultSet &result,
                             bool is_inner_sql)
{
  int ret = OB_SUCCESS;
  ParamStore *ps_ab_params = NULL;
  int get_plan_err = OB_SUCCESS;
  context.is_prepare_protocol_ = true;
  ObPsStmtId inner_stmt_id = client_stmt_id;
  context.stmt_type_ = stmt_type;

  // normal ps execute sql also a dynamic sql, here we adjust is_dynamic_sql_.
  context.is_dynamic_sql_ = !context.is_dynamic_sql_ ? !is_inner_sql : context.is_dynamic_sql_;

  ObIAllocator &allocator = result.get_mem_pool();
  ObSQLSessionInfo &session = result.get_session();
  ObExecContext &ectx = result.get_exec_context();
  ParamStore fixed_params( (ObWrapperAllocator(allocator)) );
  ParamStore ps_params( (ObWrapperAllocator(allocator)) );
  ObPsCache *ps_cache = ps_cache_;
  ObPlanCache *plan_cache = plan_cache_;
  bool use_plan_cache = session.get_local_ob_enable_plan_cache();
  ObPhysicalPlanCtx *pctx = ectx.get_physical_plan_ctx();
  ObSchemaGetterGuard *schema_guard = context.schema_guard_;
  int64_t origin_params_count = params.count();
  if (OB_ISNULL(ps_cache) || OB_ISNULL(pctx) || OB_ISNULL(schema_guard) || OB_ISNULL(plan_cache)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_ERROR("physical plan context or ps plan cache is NULL or schema_guard is null",
              K(ret), K(pctx), K(ps_cache));
  } else if (!is_inner_sql && OB_FAIL(session.get_inner_ps_stmt_id(client_stmt_id,
                                                                    inner_stmt_id))) {
  } else {
    context.statement_id_ = inner_stmt_id;
    ObPsStmtInfoGuard guard;
    ObPsStmtInfo *ps_info = NULL;
    pctx->set_original_param_cnt(origin_params_count);
    pctx->set_orig_question_mark_cnt(origin_params_count);
    if (OB_FAIL(ps_cache->get_stmt_info_guard(inner_stmt_id, guard))) {
    } else if (OB_ISNULL(ps_info = guard.get_stmt_info())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (ps_info->get_question_mark_count() != origin_params_count) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "execute");
    } else if (OB_FAIL(clac_fixed_param_store(stmt_type,
                                              ps_info->get_raw_params_idx(),
                                              ps_info->get_fixed_raw_params(),
                                              allocator,
                                              session,
                                              fixed_params))) {
    } else if (OB_FAIL(reconstruct_ps_params_store(
        allocator, context, params, fixed_params, ps_info, ps_params, ps_ab_params))) {
    } else if (context.is_batch_params_execute() &&
        OB_ISNULL(ps_ab_params)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(construct_param_store(ps_params, pctx->get_param_store_for_update()))) {
    } else {
      const ObString &sql = !ps_info->get_no_param_sql().empty() ? ps_info->get_no_param_sql() : ps_info->get_ps_sql();
      context.cur_sql_ = sql;
      context.raw_sql_ = ps_info->get_ps_sql();
#ifndef NDEBUG
      LOG_INFO("Begin to handle execute statement", "sess_id", session.get_server_sid(), K(sql));
#endif

      if (!ps_info->get_fixed_raw_params().empty()) {
        pctx->set_ps_fixed_array_index(ps_info->get_raw_params_idx());
      }
      if (OB_FAIL(session.store_query_string(sql))) {
      } else if (FALSE_IT(generate_ps_sql_id(sql, context))) {
      } else if (OB_LIKELY(ObStmt::is_dml_stmt(stmt_type))) {
        //if plan not exist, generate plan
        ObPlanCacheCtx pc_ctx(sql, PC_PS_MODE, allocator, context, ectx);
        pc_ctx.fp_result_.pc_key_.key_id_ = inner_stmt_id;
        pc_ctx.normal_parse_const_cnt_ = ps_params.count();
        context.plan_key_.db_id_ = session.get_database_id();
        pc_ctx.set_is_parameterized_execute();
        pc_ctx.set_is_inner_sql(is_inner_sql);
        pc_ctx.ab_params_ = ps_ab_params;
        if (OB_FAIL(construct_parameterized_params(ps_params, pc_ctx))) {
        } else {
          if (!use_plan_cache) {
            /*do nothing*/
          } else if (OB_FAIL(pc_get_plan_and_fill_result(pc_ctx, result, get_plan_err,
                                          ectx.get_need_disconnect_for_update()))) {
          }

          if (OB_FAIL(ret)) {//do nothing
          } else if (!result.get_is_from_plan_cache()) {
            pctx->set_original_param_cnt(origin_params_count);
            pctx->get_param_store_for_update().reset();
            // Use SMART_CALL_LARGE for complex SQL plan generation when plan cache misses
            if (OB_FAIL(SMART_CALL_LARGE(handle_physical_plan(sql, context, result, pc_ctx, get_plan_err)))) {
            }
          }
          if (OB_SUCC(ret) && (OB_FAIL(after_get_plan(pc_ctx,
                                                      session,
                                                      result.get_physical_plan(),
                                                      result.get_is_from_plan_cache())))) {
          }
        }
      } else if (stmt::T_ANONYMOUS_BLOCK == stmt_type && !context.is_mock_prepare_) {
        ParseResult parse_result;
        MEMSET(&parse_result, 0, SIZEOF(ParseResult));
        if (OB_FAIL(generate_physical_plan(parse_result, NULL, context, result,
                                            false/*is_begin_commit_stmt*/, PC_PS_MODE))) {
        } else {
          const ObPsSqlMeta &sql_meta = ps_info->get_ps_sql_meta();
          const common::ObIArray<ObField> &param_fields = sql_meta.get_param_fields();
          int64_t field_column_cnt = 0;
          for (int64_t i = 0; OB_SUCC(ret) && i < param_fields.count(); ++i) {
            if (ObRoutineParamInOut::SP_PARAM_INOUT
                  == static_cast<ObRoutineParamInOut>(param_fields.at(i).inout_mode_)
                || ObRoutineParamInOut::SP_PARAM_OUT
                  == static_cast<ObRoutineParamInOut>(param_fields.at(i).inout_mode_)) {
              field_column_cnt++;
            }
          }
          OZ (result.reserve_field_columns(field_column_cnt));
          for (int64_t i = 0; OB_SUCC(ret) && i < field_column_cnt; ++i) {
            ObField field;
            field.type_.set_type(ObNullType);
            OZ (result.add_field_column(field));
          }
        }
      } else {
        if (stmt::T_CALL_PROCEDURE == stmt_type && !context.is_dynamic_sql_) {
          // call procedure stmt call always parse as dynamic sql
          context.is_dynamic_sql_ = true;
        }
        if ((stmt::T_CALL_PROCEDURE == stmt_type || stmt::T_ANONYMOUS_BLOCK == stmt_type) && !context.is_execute_call_stmt_) {
          context.is_execute_call_stmt_ = true;
        }
        ObParser parser(allocator, session.get_sql_mode(),
                        session.get_charsets4parser());
        ParseResult parse_result;
        ObSqlTraits sql_traits;
        ParseMode parse_mode = context.is_dynamic_sql_ ? DYNAMIC_SQL_MODE :
                                (context.session_info_->is_for_trigger_package() ? TRIGGER_MODE : STD_MODE);
        if (OB_FAIL(parser.parse(sql, parse_result, parse_mode))) {
        }
        context.is_sensitive_ |= parse_result.contain_sensitive_data_;

        if (OB_FAIL(ret)) {
        } else if (OB_FAIL(check_read_only_privilege(parse_result, ectx, *schema_guard, sql_traits))) {
        } else if (OB_FAIL(generate_physical_plan(parse_result, NULL, context, result,
            false /*is_begin_commit_stmt*/, PC_PS_MODE))) {
        } // TODO Generate the path for physical plan distinction
      }
    }
  }
  return ret;
}

namespace file_sql
{

enum FileColType
{
  COL_UNKNOWN = 0,
  COL_BOOLEAN,
  COL_BIGINT,
  COL_DOUBLE,
  COL_DATE,
  COL_DATETIME,
  COL_VARCHAR
};

struct FileCell
{
  bool is_null;
  std::string val;
  FileCell() : is_null(true), val() {}
  FileCell(bool n, const std::string &v) : is_null(n), val(v) {}
};

struct FileColumn
{
  std::string name;
  FileColType type;
  bool has_null;
  FileColumn() : name(), type(COL_UNKNOWN), has_null(false) {}
};

struct FileTable
{
  std::vector<FileColumn> cols;
  std::vector<std::vector<FileCell>> rows;
};

struct NamedOption
{
  std::string key;
  std::string val;
  bool val_was_quoted;
  NamedOption() : key(), val(), val_was_quoted(false) {}
};

static std::string str_trim(const std::string &s)
{
  size_t b = 0;
  while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
    ++b;
  }
  size_t e = s.size();
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
    --e;
  }
  return s.substr(b, e - b);
}

static std::string str_tolower(const std::string &s)
{
  std::string out = s;
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(out[i])));
  }
  return out;
}

static bool is_ident_char(char c)
{
  unsigned char uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) || c == '_' || c == '$';
}

static bool is_bool_str(const std::string &s, bool &out_val)
{
  if (s.size() == 4 && 0 == STRNCASECMP(s.c_str(), "true", 4)) {
    out_val = true;
    return true;
  }
  if (s.size() == 5 && 0 == STRNCASECMP(s.c_str(), "false", 5)) {
    out_val = false;
    return true;
  }
  return false;
}

static bool is_int64_str(const std::string &s)
{
  if (s.empty()) {
    return false;
  }
  size_t idx = 0;
  if (s[0] == '+' || s[0] == '-') {
    idx = 1;
  }
  if (idx >= s.size()) {
    return false;
  }
  if (s.size() - idx > 1 && s[idx] == '0') {
    return false;
  }
  for (size_t i = idx; i < s.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  errno = 0;
  char *endp = NULL;
  (void)std::strtoll(s.c_str(), &endp, 10);
  return errno == 0 && endp == s.c_str() + s.size();
}

static bool is_double_str(const std::string &s)
{
  if (s.empty()) {
    return false;
  }
  if (std::isspace(static_cast<unsigned char>(s.front()))
      || std::isspace(static_cast<unsigned char>(s.back()))) {
    return false;
  }
  errno = 0;
  char *endp = NULL;
  double v = std::strtod(s.c_str(), &endp);
  if (errno != 0 || endp != s.c_str() + s.size() || !std::isfinite(v)) {
    return false;
  }
  return true;
}

static bool is_date_str(const std::string &s)
{
  if (s.size() != 10 || s[4] != '-' || s[7] != '-') {
    return false;
  }
  for (size_t i = 0; i < 10; ++i) {
    if (i == 4 || i == 7) {
      continue;
    }
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  int m = (s[5] - '0') * 10 + (s[6] - '0');
  int d = (s[8] - '0') * 10 + (s[9] - '0');
  return (m >= 1 && m <= 12 && d >= 1 && d <= 31);
}

static bool is_datetime_str(const std::string &s, std::string *normalized = NULL)
{
  if (s.size() < 19) {
    return false;
  }
  if (!is_date_str(s.substr(0, 10))) {
    return false;
  }
  if (s[10] != ' ' && s[10] != 'T' && s[10] != 't') {
    return false;
  }
  if (s[13] != ':' || s[16] != ':') {
    return false;
  }
  const int digit_pos[6] = {11, 12, 14, 15, 17, 18};
  for (int i = 0; i < 6; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[digit_pos[i]]))) {
      return false;
    }
  }
  int hh = (s[11] - '0') * 10 + (s[12] - '0');
  int mm = (s[14] - '0') * 10 + (s[15] - '0');
  int ss = (s[17] - '0') * 10 + (s[18] - '0');
  if (hh < 0 || hh > 23 || mm < 0 || mm > 59 || ss < 0 || ss > 59) {
    return false;
  }
  if (s.size() > 19) {
    if (s[19] != '.' || s.size() == 20 || s.size() > 26) {
      return false;
    }
    for (size_t i = 20; i < s.size(); ++i) {
      if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
        return false;
      }
    }
  }
  if (NULL != normalized) {
    *normalized = s;
    (*normalized)[10] = ' ';
  }
  return true;
}

static FileColType infer_string_token_type(const std::string &s)
{
  bool bval = false;
  if (is_bool_str(s, bval)) {
    return COL_BOOLEAN;
  }
  if (is_int64_str(s)) {
    return COL_BIGINT;
  }
  if (is_double_str(s)) {
    return COL_DOUBLE;
  }
  if (is_date_str(s)) {
    return COL_DATE;
  }
  if (is_datetime_str(s)) {
    return COL_DATETIME;
  }
  return COL_VARCHAR;
}

static FileColType merge_col_type(FileColType a, FileColType b)
{
  if (a == COL_UNKNOWN) {
    return b;
  }
  if (b == COL_UNKNOWN || a == b) {
    return a;
  }
  if ((a == COL_BIGINT && b == COL_DOUBLE) || (a == COL_DOUBLE && b == COL_BIGINT)) {
    return COL_DOUBLE;
  }
  if ((a == COL_DATE && b == COL_DATETIME) || (a == COL_DATETIME && b == COL_DATE)) {
    return COL_DATETIME;
  }
  return COL_VARCHAR;
}

static const char *col_type_to_describe_name(FileColType t)
{
  switch (t) {
    case COL_BOOLEAN: return "BOOLEAN";
    case COL_BIGINT: return "BIGINT";
    case COL_DOUBLE: return "DOUBLE";
    case COL_DATE: return "DATE";
    case COL_DATETIME: return "DATETIME";
    case COL_VARCHAR:
    default: return "VARCHAR";
  }
}

static std::string escape_sql_string(const std::string &s)
{
  std::string out;
  out.reserve(s.size() + 8);
  out.push_back('\'');
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\'') {
      out.append("''");
    } else if (c == '\\') {
      out.append("\\\\");
    } else if (c == '\0') {
      out.append("\\0");
    } else {
      out.push_back(c);
    }
  }
  out.push_back('\'');
  return out;
}

static std::string escape_sql_ident(const std::string &ident)
{
  std::string out;
  out.reserve(ident.size() + 4);
  out.push_back('`');
  for (size_t i = 0; i < ident.size(); ++i) {
    if (ident[i] == '`') {
      out.append("``");
    } else {
      out.push_back(ident[i]);
    }
  }
  out.push_back('`');
  return out;
}

static std::string format_cell_sql_expr(const FileCell &cell, FileColType col_type)
{
  if (cell.is_null) {
    switch (col_type) {
      case COL_BOOLEAN:
      case COL_BIGINT:
        return "CAST(NULL AS SIGNED)";
      case COL_DOUBLE:
        return "CAST(NULL AS DOUBLE)";
      case COL_DATE:
        return "CAST(NULL AS DATE)";
      case COL_DATETIME:
        return "CAST(NULL AS DATETIME)";
      case COL_VARCHAR:
      default:
        return "CAST(NULL AS CHAR(256))";
    }
  }
  switch (col_type) {
    case COL_BOOLEAN: {
      bool bval = false;
      if (is_bool_str(cell.val, bval)) {
        return bval ? "CAST(1 AS SIGNED)" : "CAST(0 AS SIGNED)";
      }
      return (cell.val != "0") ? "CAST(1 AS SIGNED)" : "CAST(0 AS SIGNED)";
    }
    case COL_BIGINT:
      return "CAST(" + cell.val + " AS SIGNED)";
    case COL_DOUBLE:
      return "CAST(" + cell.val + " AS DOUBLE)";
    case COL_DATE:
      return "CAST(" + escape_sql_string(cell.val) + " AS DATE)";
    case COL_DATETIME: {
      std::string norm = cell.val;
      (void)is_datetime_str(cell.val, &norm);
      return "CAST(" + escape_sql_string(norm) + " AS DATETIME)";
    }
    case COL_VARCHAR:
    default:
      return escape_sql_string(cell.val);
  }
}

static int read_entire_file(const std::string &path, std::string &out_content)
{
  int ret = OB_SUCCESS;
  out_content.clear();
  std::FILE *fp = std::fopen(path.c_str(), "rb");
  if (NULL == fp) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "file path (cannot open file)");
  } else {
    char buf[8192];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) {
      out_content.append(buf, n);
    }
    if (std::ferror(fp)) {
      ret = OB_INVALID_ARGUMENT;
    }
    std::fclose(fp);
  }
  return ret;
}

static bool unquote_sql_literal(const std::string &raw, std::string &out, bool &was_quoted)
{
  std::string s = str_trim(raw);
  was_quoted = false;
  if (s.size() >= 2 && ((s.front() == '\'' && s.back() == '\'') || (s.front() == '"' && s.back() == '"'))) {
    char q = s.front();
    out.clear();
    for (size_t i = 1; i + 1 < s.size(); ++i) {
      if (s[i] == q && i + 2 < s.size() && s[i + 1] == q) {
        out.push_back(q);
        ++i;
      } else if (s[i] == '\\' && i + 2 < s.size()) {
        char nx = s[i + 1];
        if (nx == 'n') { out.push_back('\n'); }
        else if (nx == 'r') { out.push_back('\r'); }
        else if (nx == 't') { out.push_back('\t'); }
        else { out.push_back(nx); }
        ++i;
      } else if (s[i] == q) {
        return false;
      } else {
        out.push_back(s[i]);
      }
    }
    was_quoted = true;
    return true;
  }
  out = s;
  return true;
}

static int parse_table_func_args(const std::string &args_str,
                                 std::string &out_path,
                                 std::vector<NamedOption> &out_opts)
{
  int ret = OB_SUCCESS;
  out_path.clear();
  out_opts.clear();
  std::vector<std::string> parts;
  size_t cur_start = 0;
  char in_quote = 0;
  int paren_depth = 0;
  for (size_t i = 0; i < args_str.size(); ++i) {
    char c = args_str[i];
    if (in_quote != 0) {
      if (c == '\\' && i + 1 < args_str.size()) {
        ++i;
      } else if (c == in_quote) {
        if (i + 1 < args_str.size() && args_str[i + 1] == in_quote) {
          ++i;
        } else {
          in_quote = 0;
        }
      }
    } else {
      if (c == '\'' || c == '"') {
        in_quote = c;
      } else if (c == '(') {
        ++paren_depth;
      } else if (c == ')') {
        if (paren_depth > 0) {
          --paren_depth;
        }
      } else if (c == ',' && paren_depth == 0) {
        parts.push_back(str_trim(args_str.substr(cur_start, i - cur_start)));
        cur_start = i + 1;
      }
    }
  }
  if (in_quote != 0 || paren_depth != 0) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(args_str.size()), args_str.c_str(), 1);
    return ret;
  }
  std::string tail = str_trim(args_str.substr(cur_start));
  if (!tail.empty() || !parts.empty()) {
    parts.push_back(tail);
  }
  if (parts.empty() || parts[0].empty()) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(args_str.size()), args_str.c_str(), 1);
    return ret;
  }
  bool first_quoted = false;
  if (!unquote_sql_literal(parts[0], out_path, first_quoted) || !first_quoted || parts[0].find('=') != std::string::npos && !first_quoted) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(parts[0].size()), parts[0].c_str(), 1);
    return ret;
  }
  for (size_t i = 1; i < parts.size(); ++i) {
    const std::string &p = parts[i];
    if (p.empty()) {
      ret = OB_ERR_PARSE_SQL;
      LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX), 0, "", 1);
      return ret;
    }
    size_t eq_pos = std::string::npos;
    char q = 0;
    for (size_t k = 0; k < p.size(); ++k) {
      if (q != 0) {
        if (p[k] == '\\' && k + 1 < p.size()) {
          ++k;
        } else if (p[k] == q) {
          if (k + 1 < p.size() && p[k + 1] == q) {
            ++k;
          } else {
            q = 0;
          }
        }
      } else {
        if (p[k] == '\'' || p[k] == '"') {
          q = p[k];
        } else if (p[k] == '=') {
          eq_pos = k;
          break;
        }
      }
    }
    if (eq_pos == std::string::npos) {
      ret = OB_ERR_PARSE_SQL;
      LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                     static_cast<int>(p.size()), p.c_str(), 1);
      return ret;
    }
    std::string k_str = str_tolower(str_trim(p.substr(0, eq_pos)));
    std::string v_raw = str_trim(p.substr(eq_pos + 1));
    if (k_str.empty() || v_raw.empty()) {
      ret = OB_ERR_PARSE_SQL;
      LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                     static_cast<int>(p.size()), p.c_str(), 1);
      return ret;
    }
    for (size_t c_idx = 0; c_idx < k_str.size(); ++c_idx) {
      if (!is_ident_char(k_str[c_idx])) {
        ret = OB_ERR_PARSE_SQL;
        LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                       static_cast<int>(p.size()), p.c_str(), 1);
        return ret;
      }
    }
    NamedOption opt;
    opt.key = k_str;
    if (!unquote_sql_literal(v_raw, opt.val, opt.val_was_quoted)) {
      ret = OB_ERR_PARSE_SQL;
      LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                     static_cast<int>(v_raw.size()), v_raw.c_str(), 1);
      return ret;
    }
    out_opts.push_back(opt);
  }
  return ret;
}

struct CsvFieldToken
{
  std::string val;
  bool was_quoted;
  CsvFieldToken() : val(), was_quoted(false) {}
};

static int read_csv_file(const std::string &path,
                         const std::vector<NamedOption> &opts,
                         FileTable &table)
{
  int ret = OB_SUCCESS;
  bool has_header = true;
  char delim = ',';
  char quote = '"';
  std::string nullstr = "";
  bool has_custom_nullstr = false;

  for (size_t i = 0; i < opts.size(); ++i) {
    const std::string &k = opts[i].key;
    const std::string &v = opts[i].val;
    std::string v_low = str_tolower(v);
    if (k == "header") {
      if (v_low == "true" || v_low == "1") {
        has_header = true;
      } else if (v_low == "false" || v_low == "0") {
        has_header = false;
      } else {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "invalid header option in read_csv");
        return ret;
      }
    } else if (k == "sample_size") {
      if (v != "-1") {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "sample_size option other than -1 in read_csv");
        return ret;
      }
    } else if (k == "delim" || k == "sep" || k == "delimiter") {
      if (v.size() != 1) {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "multi-character delimiter in read_csv");
        return ret;
      }
      delim = v[0];
    } else if (k == "quote") {
      if (v.size() != 1) {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "multi-character quote in read_csv");
        return ret;
      }
      quote = v[0];
    } else if (k == "nullstr" || k == "null") {
      nullstr = v;
      has_custom_nullstr = true;
    } else {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported option in read_csv");
      return ret;
    }
  }

  std::string content;
  if (OB_FAIL(read_entire_file(path, content))) {
    return ret;
  }
  size_t pos = 0;
  if (content.size() >= 3
      && static_cast<unsigned char>(content[0]) == 0xEF
      && static_cast<unsigned char>(content[1]) == 0xBB
      && static_cast<unsigned char>(content[2]) == 0xBF) {
    pos = 3;
  }

  std::vector<std::vector<CsvFieldToken>> raw_rows;
  std::vector<CsvFieldToken> cur_row;
  CsvFieldToken cur_field;
  bool in_quotes = false;
  bool at_field_start = true;

  while (pos < content.size()) {
    char c = content[pos];
    if (in_quotes) {
      if (c == quote) {
        if (pos + 1 < content.size() && content[pos + 1] == quote) {
          cur_field.val.push_back(quote);
          pos += 2;
        } else {
          in_quotes = false;
          ++pos;
        }
      } else if (c == '\r') {
        if (pos + 1 < content.size() && content[pos + 1] == '\n') {
          ++pos;
        }
        cur_field.val.push_back('\n');
        ++pos;
      } else {
        cur_field.val.push_back(c);
        ++pos;
      }
    } else {
      if (at_field_start && c == quote) {
        in_quotes = true;
        cur_field.was_quoted = true;
        at_field_start = false;
        ++pos;
      } else if (c == delim) {
        cur_row.push_back(cur_field);
        cur_field = CsvFieldToken();
        at_field_start = true;
        ++pos;
      } else if (c == '\r' || c == '\n') {
        if (c == '\r' && pos + 1 < content.size() && content[pos + 1] == '\n') {
          ++pos;
        }
        cur_row.push_back(cur_field);
        raw_rows.push_back(cur_row);
        cur_row.clear();
        cur_field = CsvFieldToken();
        at_field_start = true;
        ++pos;
      } else {
        cur_field.val.push_back(c);
        at_field_start = false;
        ++pos;
      }
    }
  }
  if (in_quotes) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "unterminated quoted field in CSV");
    return ret;
  }
  if (!cur_row.empty() || !cur_field.val.empty() || cur_field.was_quoted) {
    cur_row.push_back(cur_field);
    raw_rows.push_back(cur_row);
  }
  if (raw_rows.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "empty CSV file");
    return ret;
  }

  size_t num_cols = raw_rows[0].size();
  if (num_cols == 0) {
    ret = OB_INVALID_ARGUMENT;
    return ret;
  }
  table.cols.resize(num_cols);
  size_t data_start_idx = 0;
  if (has_header) {
    data_start_idx = 1;
    std::map<std::string, int> name_counts;
    for (size_t j = 0; j < num_cols; ++j) {
      std::string base = raw_rows[0][j].val;
      if (base.empty()) {
        base = "column" + std::to_string(j);
      }
      int cnt = ++name_counts[base];
      std::string final_name = base;
      if (cnt > 1) {
        final_name = base + "__" + std::to_string(cnt);
      }
      table.cols[j].name = final_name;
    }
  } else {
    for (size_t j = 0; j < num_cols; ++j) {
      table.cols[j].name = "column" + std::to_string(j);
    }
  }

  for (size_t r = data_start_idx; r < raw_rows.size(); ++r) {
    const std::vector<CsvFieldToken> &src_row = raw_rows[r];
    if (src_row.size() == 1 && src_row[0].val.empty() && !src_row[0].was_quoted && num_cols > 1) {
      continue;
    }
    std::vector<FileCell> dst_row(num_cols);
    for (size_t j = 0; j < num_cols; ++j) {
      if (j >= src_row.size()) {
        dst_row[j].is_null = true;
        table.cols[j].has_null = true;
      } else {
        const CsvFieldToken &tok = src_row[j];
        bool is_null_cell = false;
        if (!tok.was_quoted) {
          if (tok.val.empty() || tok.val == "NULL") {
            is_null_cell = true;
          } else if (has_custom_nullstr && tok.val == nullstr) {
            is_null_cell = true;
          }
        }
        if (is_null_cell) {
          dst_row[j].is_null = true;
          table.cols[j].has_null = true;
        } else {
          dst_row[j].is_null = false;
          dst_row[j].val = tok.val;
          FileColType t = infer_string_token_type(tok.val);
          table.cols[j].type = merge_col_type(table.cols[j].type, t);
        }
      }
    }
    table.rows.push_back(dst_row);
  }

  for (size_t j = 0; j < num_cols; ++j) {
    if (table.cols[j].type == COL_UNKNOWN) {
      table.cols[j].type = COL_VARCHAR;
    }
    if (table.rows.empty()) {
      table.cols[j].has_null = true;
    }
  }
  return ret;
}

struct SimpleJsonValue
{
  enum Kind { J_NULL, J_BOOL, J_INT, J_DOUBLE, J_STR, J_OBJ, J_ARR };
  Kind kind;
  std::string str_val;
  std::vector<std::pair<std::string, SimpleJsonValue>> obj_fields;
  SimpleJsonValue() : kind(J_NULL), str_val(), obj_fields() {}
};

class SimpleJsonParser
{
public:
  SimpleJsonParser(const char *p, size_t len) : p_(p), len_(len), pos_(0) {}

  void skip_ws()
  {
    while (pos_ < len_ && std::isspace(static_cast<unsigned char>(p_[pos_]))) {
      ++pos_;
    }
  }

  bool eof() { skip_ws(); return pos_ >= len_; }

  bool parse_value(SimpleJsonValue &out)
  {
    skip_ws();
    if (pos_ >= len_) {
      return false;
    }
    char c = p_[pos_];
    if (c == '{') {
      return parse_object(out);
    } else if (c == '[') {
      return parse_array(out);
    } else if (c == '"') {
      out.kind = SimpleJsonValue::J_STR;
      return parse_string(out.str_val);
    } else if (c == 't') {
      if (pos_ + 4 <= len_ && 0 == std::memcmp(p_ + pos_, "true", 4)) {
        pos_ += 4;
        out.kind = SimpleJsonValue::J_BOOL;
        out.str_val = "true";
        return true;
      }
      return false;
    } else if (c == 'f') {
      if (pos_ + 5 <= len_ && 0 == std::memcmp(p_ + pos_, "false", 5)) {
        pos_ += 5;
        out.kind = SimpleJsonValue::J_BOOL;
        out.str_val = "false";
        return true;
      }
      return false;
    } else if (c == 'n') {
      if (pos_ + 4 <= len_ && 0 == std::memcmp(p_ + pos_, "null", 4)) {
        pos_ += 4;
        out.kind = SimpleJsonValue::J_NULL;
        return true;
      }
      return false;
    } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
      return parse_number(out);
    }
    return false;
  }

  bool parse_string(std::string &out)
  {
    out.clear();
    if (pos_ >= len_ || p_[pos_] != '"') {
      return false;
    }
    ++pos_;
    while (pos_ < len_) {
      char c = p_[pos_++];
      if (c == '"') {
        return true;
      } else if (c == '\\') {
        if (pos_ >= len_) {
          return false;
        }
        char esc = p_[pos_++];
        switch (esc) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (pos_ + 4 > len_) {
              return false;
            }
            unsigned int cp = 0;
            for (int k = 0; k < 4; ++k) {
              char hc = p_[pos_++];
              cp <<= 4;
              if (hc >= '0' && hc <= '9') cp |= (hc - '0');
              else if (hc >= 'a' && hc <= 'f') cp |= (hc - 'a' + 10);
              else if (hc >= 'A' && hc <= 'F') cp |= (hc - 'A' + 10);
              else return false;
            }
            if (cp <= 0x7F) {
              out.push_back(static_cast<char>(cp));
            } else if (cp <= 0x7FF) {
              out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default:
            return false;
        }
      } else if (static_cast<unsigned char>(c) < 0x20) {
        return false;
      } else {
        out.push_back(c);
      }
    }
    return false;
  }

  bool parse_number(SimpleJsonValue &out)
  {
    size_t start = pos_;
    bool is_float = false;
    if (p_[pos_] == '-') {
      ++pos_;
    }
    if (pos_ >= len_ || !std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
      return false;
    }
    if (p_[pos_] == '0') {
      ++pos_;
    } else {
      while (pos_ < len_ && std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < len_ && p_[pos_] == '.') {
      is_float = true;
      ++pos_;
      if (pos_ >= len_ || !std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
        return false;
      }
      while (pos_ < len_ && std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
        ++pos_;
      }
    }
    if (pos_ < len_ && (p_[pos_] == 'e' || p_[pos_] == 'E')) {
      is_float = true;
      ++pos_;
      if (pos_ < len_ && (p_[pos_] == '+' || p_[pos_] == '-')) {
        ++pos_;
      }
      if (pos_ >= len_ || !std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
        return false;
      }
      while (pos_ < len_ && std::isdigit(static_cast<unsigned char>(p_[pos_]))) {
        ++pos_;
      }
    }
    out.str_val.assign(p_ + start, pos_ - start);
    if (!is_float && is_int64_str(out.str_val)) {
      out.kind = SimpleJsonValue::J_INT;
    } else if (is_double_str(out.str_val)) {
      out.kind = SimpleJsonValue::J_DOUBLE;
    } else {
      return false;
    }
    return true;
  }

  bool parse_object(SimpleJsonValue &out)
  {
    out.kind = SimpleJsonValue::J_OBJ;
    out.obj_fields.clear();
    size_t start = pos_;
    ++pos_; // skip '{'
    skip_ws();
    if (pos_ < len_ && p_[pos_] == '}') {
      ++pos_;
      out.str_val.assign(p_ + start, pos_ - start);
      return true;
    }
    while (pos_ < len_) {
      skip_ws();
      std::string key;
      if (!parse_string(key)) {
        return false;
      }
      skip_ws();
      if (pos_ >= len_ || p_[pos_] != ':') {
        return false;
      }
      ++pos_;
      SimpleJsonValue val;
      if (!parse_value(val)) {
        return false;
      }
      out.obj_fields.push_back(std::make_pair(key, val));
      skip_ws();
      if (pos_ < len_ && p_[pos_] == ',') {
        ++pos_;
      } else if (pos_ < len_ && p_[pos_] == '}') {
        ++pos_;
        out.str_val.assign(p_ + start, pos_ - start);
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

  bool parse_array(SimpleJsonValue &out)
  {
    out.kind = SimpleJsonValue::J_ARR;
    size_t start = pos_;
    ++pos_; // skip '['
    skip_ws();
    if (pos_ < len_ && p_[pos_] == ']') {
      ++pos_;
      out.str_val.assign(p_ + start, pos_ - start);
      return true;
    }
    while (pos_ < len_) {
      SimpleJsonValue elem;
      if (!parse_value(elem)) {
        return false;
      }
      skip_ws();
      if (pos_ < len_ && p_[pos_] == ',') {
        ++pos_;
      } else if (pos_ < len_ && p_[pos_] == ']') {
        ++pos_;
        out.str_val.assign(p_ + start, pos_ - start);
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

private:
  const char *p_;
  size_t len_;
  size_t pos_;
};

static int read_json_file(const std::string &path,
                          const std::vector<NamedOption> &opts,
                          FileTable &table)
{
  int ret = OB_SUCCESS;
  std::string format = "auto";
  for (size_t i = 0; i < opts.size(); ++i) {
    const std::string &k = opts[i].key;
    std::string v_low = str_tolower(opts[i].val);
    if (k == "format") {
      if (v_low == "newline_delimited" || v_low == "ndjson" || v_low == "auto" || v_low == "array") {
        format = v_low;
      } else {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported format in read_json");
        return ret;
      }
    } else if (k == "sample_size") {
      if (opts[i].val != "-1") {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "sample_size option other than -1 in read_json");
        return ret;
      }
    } else {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported option in read_json");
      return ret;
    }
  }

  std::string content;
  if (OB_FAIL(read_entire_file(path, content))) {
    return ret;
  }

  std::vector<SimpleJsonValue> objects;
  size_t line_start = 0;
  while (line_start < content.size()) {
    size_t line_end = content.find('\n', line_start);
    if (line_end == std::string::npos) {
      line_end = content.size();
    }
    std::string line = str_trim(content.substr(line_start, line_end - line_start));
    line_start = line_end + 1;
    if (line.empty()) {
      continue;
    }
    SimpleJsonParser parser(line.c_str(), line.size());
    SimpleJsonValue jval;
    if (!parser.parse_value(jval) || !parser.eof() || jval.kind != SimpleJsonValue::J_OBJ) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "invalid JSON object line");
      return ret;
    }
    objects.push_back(jval);
  }

  if (objects.empty()) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "empty JSON file");
    return ret;
  }

  std::map<std::string, size_t> col_idx_map;
  for (size_t r = 0; r < objects.size(); ++r) {
    const auto &fields = objects[r].obj_fields;
    for (size_t f = 0; f < fields.size(); ++f) {
      const std::string &k = fields[f].first;
      if (col_idx_map.find(k) == col_idx_map.end()) {
        size_t idx = table.cols.size();
        col_idx_map[k] = idx;
        FileColumn col;
        col.name = k;
        table.cols.push_back(col);
      }
    }
  }

  size_t num_cols = table.cols.size();
  if (num_cols == 0) {
    ret = OB_INVALID_ARGUMENT;
    return ret;
  }

  for (size_t r = 0; r < objects.size(); ++r) {
    std::vector<FileCell> row(num_cols);
    std::vector<bool> seen(num_cols, false);
    const auto &fields = objects[r].obj_fields;
    for (size_t f = 0; f < fields.size(); ++f) {
      size_t cidx = col_idx_map[fields[f].first];
      seen[cidx] = true;
      const SimpleJsonValue &v = fields[f].second;
      if (v.kind == SimpleJsonValue::J_NULL) {
        row[cidx].is_null = true;
        table.cols[cidx].has_null = true;
      } else {
        row[cidx].is_null = false;
        row[cidx].val = v.str_val;
        FileColType t = COL_VARCHAR;
        if (v.kind == SimpleJsonValue::J_BOOL) {
          t = COL_BOOLEAN;
        } else if (v.kind == SimpleJsonValue::J_INT) {
          t = COL_BIGINT;
        } else if (v.kind == SimpleJsonValue::J_DOUBLE) {
          t = COL_DOUBLE;
        } else if (v.kind == SimpleJsonValue::J_STR) {
          if (is_date_str(v.str_val)) {
            t = COL_DATE;
          } else if (is_datetime_str(v.str_val)) {
            t = COL_DATETIME;
          } else {
            t = COL_VARCHAR;
          }
        }
        table.cols[cidx].type = merge_col_type(table.cols[cidx].type, t);
      }
    }
    for (size_t j = 0; j < num_cols; ++j) {
      if (!seen[j]) {
        row[j].is_null = true;
        table.cols[j].has_null = true;
      }
    }
    table.rows.push_back(row);
  }

  for (size_t j = 0; j < num_cols; ++j) {
    if (table.cols[j].type == COL_UNKNOWN) {
      table.cols[j].type = COL_VARCHAR;
    }
  }
  return ret;
}

class ThriftCompactReader
{
public:
  ThriftCompactReader(const uint8_t *buf, size_t len) : buf_(buf), len_(len), pos_(0), ok_(true) {}

  bool ok() const { return ok_; }
  size_t pos() const { return pos_; }

  uint8_t read_u8()
  {
    if (pos_ >= len_) {
      ok_ = false;
      return 0;
    }
    return buf_[pos_++];
  }

  uint64_t read_varint()
  {
    uint64_t result = 0;
    int shift = 0;
    while (pos_ < len_) {
      uint8_t b = buf_[pos_++];
      result |= static_cast<uint64_t>(b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        return result;
      }
      shift += 7;
      if (shift >= 64) {
        ok_ = false;
        return 0;
      }
    }
    ok_ = false;
    return 0;
  }

  int64_t read_zigzag()
  {
    uint64_t n = read_varint();
    return static_cast<int64_t>((n >> 1) ^ (~(n & 1) + 1));
  }

  std::string read_binary()
  {
    uint64_t slen = read_varint();
    if (!ok_ || pos_ + slen > len_) {
      ok_ = false;
      return "";
    }
    std::string s(reinterpret_cast<const char *>(buf_ + pos_), static_cast<size_t>(slen));
    pos_ += static_cast<size_t>(slen);
    return s;
  }

  bool read_field_header(int16_t &last_field_id, int16_t &field_id, uint8_t &field_type)
  {
    uint8_t b = read_u8();
    if (!ok_ || b == 0) {
      field_type = 0;
      return false;
    }
    field_type = b & 0x0F;
    uint8_t delta = (b >> 4) & 0x0F;
    if (delta == 0) {
      field_id = static_cast<int16_t>(read_zigzag());
    } else {
      field_id = static_cast<int16_t>(last_field_id + delta);
    }
    last_field_id = field_id;
    return ok_;
  }

  bool read_list_header(uint8_t &elem_type, uint64_t &count)
  {
    uint8_t b = read_u8();
    if (!ok_) {
      return false;
    }
    elem_type = b & 0x0F;
    uint8_t hi = (b >> 4) & 0x0F;
    if (hi == 15) {
      count = read_varint();
    } else {
      count = hi;
    }
    return ok_;
  }

  void skip_value(uint8_t type)
  {
    if (!ok_) {
      return;
    }
    switch (type) {
      case 1:
      case 2:
        break;
      case 3:
        (void)read_u8();
        break;
      case 4:
      case 5:
      case 6:
        (void)read_varint();
        break;
      case 7:
        if (pos_ + 8 > len_) { ok_ = false; } else { pos_ += 8; }
        break;
      case 8:
        (void)read_binary();
        break;
      case 9:
      case 10: {
        uint8_t et = 0;
        uint64_t cnt = 0;
        if (read_list_header(et, cnt)) {
          for (uint64_t i = 0; i < cnt && ok_; ++i) {
            skip_value(et);
          }
        }
        break;
      }
      case 11: {
        uint64_t cnt = read_varint();
        if (cnt > 0 && ok_) {
          uint8_t kv = read_u8();
          uint8_t kt = (kv >> 4) & 0x0F;
          uint8_t vt = kv & 0x0F;
          for (uint64_t i = 0; i < cnt && ok_; ++i) {
            skip_value(kt);
            skip_value(vt);
          }
        }
        break;
      }
      case 12: {
        int16_t last_fid = 0;
        int16_t fid = 0;
        uint8_t ft = 0;
        while (read_field_header(last_fid, fid, ft)) {
          skip_value(ft);
        }
        break;
      }
      default:
        ok_ = false;
        break;
    }
  }

private:
  const uint8_t *buf_;
  size_t len_;
  size_t pos_;
  bool ok_;
};

static bool snappy_decompress(const uint8_t *src, size_t src_len, std::vector<uint8_t> &dst)
{
  size_t pos = 0;
  uint64_t uncompressed_len = 0;
  int shift = 0;
  while (pos < src_len) {
    uint8_t b = src[pos++];
    uncompressed_len |= static_cast<uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) {
      break;
    }
    shift += 7;
    if (shift >= 32) {
      return false;
    }
  }
  if (uncompressed_len > 256ULL * 1024ULL * 1024ULL) {
    return false;
  }
  dst.clear();
  dst.reserve(static_cast<size_t>(uncompressed_len));
  while (pos < src_len) {
    uint8_t tag = src[pos++];
    uint8_t tag_type = tag & 0x03;
    if (tag_type == 0) {
      uint32_t len = (tag >> 2) + 1;
      if ((tag >> 2) >= 60) {
        uint32_t extra_bytes = (tag >> 2) - 59;
        if (pos + extra_bytes > src_len) {
          return false;
        }
        len = 0;
        for (uint32_t k = 0; k < extra_bytes; ++k) {
          len |= static_cast<uint32_t>(src[pos++]) << (8 * k);
        }
        len += 1;
      }
      if (pos + len > src_len) {
        return false;
      }
      dst.insert(dst.end(), src + pos, src + pos + len);
      pos += len;
    } else {
      uint32_t len = 0;
      uint32_t offset = 0;
      if (tag_type == 1) {
        if (pos >= src_len) {
          return false;
        }
        len = ((tag >> 2) & 0x07) + 4;
        offset = (static_cast<uint32_t>(tag & 0xE0) << 3) | src[pos++];
      } else if (tag_type == 2) {
        if (pos + 2 > src_len) {
          return false;
        }
        len = (tag >> 2) + 1;
        offset = static_cast<uint32_t>(src[pos]) | (static_cast<uint32_t>(src[pos + 1]) << 8);
        pos += 2;
      } else {
        if (pos + 4 > src_len) {
          return false;
        }
        len = (tag >> 2) + 1;
        offset = static_cast<uint32_t>(src[pos])
               | (static_cast<uint32_t>(src[pos + 1]) << 8)
               | (static_cast<uint32_t>(src[pos + 2]) << 16)
               | (static_cast<uint32_t>(src[pos + 3]) << 24);
        pos += 4;
      }
      if (offset == 0 || offset > dst.size()) {
        return false;
      }
      size_t copy_pos = dst.size() - offset;
      for (uint32_t k = 0; k < len; ++k) {
        dst.push_back(dst[copy_pos + k]);
      }
    }
  }
  return dst.size() == uncompressed_len;
}

static bool decode_rle_bit_packed(const uint8_t *buf,
                                  size_t len,
                                  uint8_t bit_width,
                                  size_t num_values,
                                  std::vector<uint32_t> &out)
{
  out.clear();
  out.reserve(num_values);
  if (bit_width == 0) {
    out.assign(num_values, 0);
    return true;
  }
  size_t pos = 0;
  uint32_t byte_width = (bit_width + 7) / 8;
  while (out.size() < num_values && pos < len) {
    uint64_t header = 0;
    int shift = 0;
    while (pos < len) {
      uint8_t b = buf[pos++];
      header |= static_cast<uint64_t>(b & 0x7F) << shift;
      if ((b & 0x80) == 0) {
        break;
      }
      shift += 7;
    }
    if ((header & 1) == 0) {
      uint64_t run_len = header >> 1;
      if (pos + byte_width > len) {
        return false;
      }
      uint32_t val = 0;
      for (uint32_t k = 0; k < byte_width; ++k) {
        val |= static_cast<uint32_t>(buf[pos++]) << (8 * k);
      }
      for (uint64_t i = 0; i < run_len && out.size() < num_values; ++i) {
        out.push_back(val);
      }
    } else {
      uint64_t num_groups = header >> 1;
      uint64_t total_vals = num_groups * 8;
      uint64_t total_bytes = num_groups * bit_width;
      if (pos + total_bytes > len) {
        return false;
      }
      uint64_t bit_buf = 0;
      int bits_in_buf = 0;
      size_t group_end = pos + static_cast<size_t>(total_bytes);
      uint32_t mask = (bit_width == 32) ? 0xFFFFFFFFU : ((1U << bit_width) - 1U);
      for (uint64_t i = 0; i < total_vals && out.size() < num_values; ++i) {
        while (bits_in_buf < bit_width && pos < group_end) {
          bit_buf |= static_cast<uint64_t>(buf[pos++]) << bits_in_buf;
          bits_in_buf += 8;
        }
        uint32_t val = static_cast<uint32_t>(bit_buf) & mask;
        bit_buf >>= bit_width;
        bits_in_buf -= bit_width;
        out.push_back(val);
      }
      pos = group_end;
    }
  }
  return out.size() == num_values;
}

struct ParquetSchemaElem
{
  int32_t type;
  int32_t repetition_type;
  std::string name;
  int32_t num_children;
  bool has_type;
  ParquetSchemaElem() : type(-1), repetition_type(0), name(), num_children(0), has_type(false) {}
};

struct ParquetColumnChunkMeta
{
  int32_t type;
  int32_t codec;
  int64_t num_values;
  int64_t data_page_offset;
  int64_t dict_page_offset;
  bool has_dict_offset;
  ParquetColumnChunkMeta()
    : type(-1), codec(0), num_values(0), data_page_offset(0), dict_page_offset(0), has_dict_offset(false) {}
};

static int read_parquet_file(const std::string &path,
                             const std::vector<NamedOption> &opts,
                             FileTable &table)
{
  int ret = OB_SUCCESS;
  if (!opts.empty()) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported option in read_parquet");
    return ret;
  }
  std::string content;
  if (OB_FAIL(read_entire_file(path, content))) {
    return ret;
  }
  if (content.size() < 12
      || 0 != std::memcmp(content.data(), "PAR1", 4)
      || 0 != std::memcmp(content.data() + content.size() - 4, "PAR1", 4)) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "invalid Parquet file magic");
    return ret;
  }
  const uint8_t *raw = reinterpret_cast<const uint8_t *>(content.data());
  size_t file_len = content.size();
  uint32_t footer_len = static_cast<uint32_t>(raw[file_len - 8])
                      | (static_cast<uint32_t>(raw[file_len - 7]) << 8)
                      | (static_cast<uint32_t>(raw[file_len - 6]) << 16)
                      | (static_cast<uint32_t>(raw[file_len - 5]) << 24);
  if (footer_len == 0 || static_cast<size_t>(footer_len) + 12 > file_len) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "invalid Parquet footer length");
    return ret;
  }

  ThriftCompactReader footer(raw + file_len - 8 - footer_len, footer_len);
  std::vector<ParquetSchemaElem> schema_elems;
  int64_t total_rows = 0;
  std::vector<std::vector<ParquetColumnChunkMeta>> row_groups;

  int16_t last_fid = 0;
  int16_t fid = 0;
  uint8_t ftype = 0;
  while (footer.read_field_header(last_fid, fid, ftype)) {
    if (fid == 2 && ftype == 9) {
      uint8_t et = 0;
      uint64_t cnt = 0;
      if (footer.read_list_header(et, cnt)) {
        for (uint64_t i = 0; i < cnt && footer.ok(); ++i) {
          ParquetSchemaElem se;
          int16_t se_last = 0, se_fid = 0;
          uint8_t se_ft = 0;
          while (footer.read_field_header(se_last, se_fid, se_ft)) {
            if (se_fid == 1 && se_ft == 5) {
              se.type = static_cast<int32_t>(footer.read_zigzag());
              se.has_type = true;
            } else if (se_fid == 3 && se_ft == 5) {
              se.repetition_type = static_cast<int32_t>(footer.read_zigzag());
            } else if (se_fid == 4 && se_ft == 8) {
              se.name = footer.read_binary();
            } else if (se_fid == 5 && se_ft == 5) {
              se.num_children = static_cast<int32_t>(footer.read_zigzag());
            } else {
              footer.skip_value(se_ft);
            }
          }
          schema_elems.push_back(se);
        }
      }
    } else if (fid == 3 && ftype == 6) {
      total_rows = footer.read_zigzag();
    } else if (fid == 4 && ftype == 9) {
      uint8_t et = 0;
      uint64_t rg_cnt = 0;
      if (footer.read_list_header(et, rg_cnt)) {
        for (uint64_t r = 0; r < rg_cnt && footer.ok(); ++r) {
          std::vector<ParquetColumnChunkMeta> rg_cols;
          int16_t rg_last = 0, rg_fid = 0;
          uint8_t rg_ft = 0;
          while (footer.read_field_header(rg_last, rg_fid, rg_ft)) {
            if (rg_fid == 1 && rg_ft == 9) {
              uint8_t cc_et = 0;
              uint64_t cc_cnt = 0;
              if (footer.read_list_header(cc_et, cc_cnt)) {
                for (uint64_t c = 0; c < cc_cnt && footer.ok(); ++c) {
                  ParquetColumnChunkMeta cmeta;
                  int16_t cc_last = 0, cc_fid = 0;
                  uint8_t cc_ft = 0;
                  while (footer.read_field_header(cc_last, cc_fid, cc_ft)) {
                    if (cc_fid == 3 && cc_ft == 12) {
                      int16_t md_last = 0, md_fid = 0;
                      uint8_t md_ft = 0;
                      while (footer.read_field_header(md_last, md_fid, md_ft)) {
                        if (md_fid == 1 && md_ft == 5) {
                          cmeta.type = static_cast<int32_t>(footer.read_zigzag());
                        } else if (md_fid == 4 && md_ft == 5) {
                          cmeta.codec = static_cast<int32_t>(footer.read_zigzag());
                        } else if (md_fid == 5 && md_ft == 6) {
                          cmeta.num_values = footer.read_zigzag();
                        } else if (md_fid == 9 && md_ft == 6) {
                          cmeta.data_page_offset = footer.read_zigzag();
                        } else if (md_fid == 11 && md_ft == 6) {
                          cmeta.dict_page_offset = footer.read_zigzag();
                          cmeta.has_dict_offset = true;
                        } else {
                          footer.skip_value(md_ft);
                        }
                      }
                    } else {
                      footer.skip_value(cc_ft);
                    }
                  }
                  rg_cols.push_back(cmeta);
                }
              }
            } else {
              footer.skip_value(rg_ft);
            }
          }
          row_groups.push_back(rg_cols);
        }
      }
    } else {
      footer.skip_value(ftype);
    }
  }

  if (!footer.ok() || schema_elems.size() < 2) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "invalid Parquet schema metadata");
    return ret;
  }

  std::vector<ParquetSchemaElem> leaf_cols;
  for (size_t i = 1; i < schema_elems.size(); ++i) {
    if (schema_elems[i].has_type && schema_elems[i].num_children == 0) {
      leaf_cols.push_back(schema_elems[i]);
    }
  }
  if (leaf_cols.empty()) {
    ret = OB_INVALID_ARGUMENT;
    return ret;
  }

  size_t num_cols = leaf_cols.size();
  table.cols.resize(num_cols);
  for (size_t j = 0; j < num_cols; ++j) {
    table.cols[j].name = leaf_cols[j].name;
    switch (leaf_cols[j].type) {
      case 0: table.cols[j].type = COL_BOOLEAN; break;
      case 1:
      case 2: table.cols[j].type = COL_BIGINT; break;
      case 4:
      case 5: table.cols[j].type = COL_DOUBLE; break;
      case 6:
      case 7:
      default: table.cols[j].type = COL_VARCHAR; break;
    }
  }

  table.rows.assign(static_cast<size_t>(total_rows), std::vector<FileCell>(num_cols));
  for (size_t j = 0; j < num_cols; ++j) {
    size_t row_cursor = 0;
    for (size_t rg = 0; rg < row_groups.size(); ++rg) {
      if (j >= row_groups[rg].size()) {
        ret = OB_INVALID_ARGUMENT;
        return ret;
      }
      const ParquetColumnChunkMeta &cmeta = row_groups[rg][j];
      size_t page_pos = static_cast<size_t>(cmeta.data_page_offset);
      if (cmeta.has_dict_offset && cmeta.dict_page_offset > 0 && cmeta.dict_page_offset < cmeta.data_page_offset) {
        page_pos = static_cast<size_t>(cmeta.dict_page_offset);
      }
      int64_t values_decoded = 0;
      std::vector<std::string> dict_vals;
      while (values_decoded < cmeta.num_values && page_pos < file_len) {
        ThriftCompactReader ph(raw + page_pos, file_len - page_pos);
        int32_t page_type = -1;
        int32_t uncompressed_size = 0;
        int32_t compressed_size = 0;
        int32_t dp_num_values = 0;
        int32_t dp_encoding = 0;
        int32_t dp2_def_len = 0;
        int32_t dp2_rep_len = 0;
        bool dp2_is_compressed = true;
        int16_t ph_last = 0, ph_fid = 0;
        uint8_t ph_ft = 0;
        while (ph.read_field_header(ph_last, ph_fid, ph_ft)) {
          if (ph_fid == 1 && ph_ft == 5) {
            page_type = static_cast<int32_t>(ph.read_zigzag());
          } else if (ph_fid == 2 && ph_ft == 5) {
            uncompressed_size = static_cast<int32_t>(ph.read_zigzag());
          } else if (ph_fid == 3 && ph_ft == 5) {
            compressed_size = static_cast<int32_t>(ph.read_zigzag());
          } else if (ph_fid == 5 && ph_ft == 12) {
            int16_t d_last = 0, d_fid = 0;
            uint8_t d_ft = 0;
            while (ph.read_field_header(d_last, d_fid, d_ft)) {
              if (d_fid == 1 && d_ft == 5) dp_num_values = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 2 && d_ft == 5) dp_encoding = static_cast<int32_t>(ph.read_zigzag());
              else ph.skip_value(d_ft);
            }
          } else if (ph_fid == 7 && ph_ft == 12) {
            int16_t d_last = 0, d_fid = 0;
            uint8_t d_ft = 0;
            while (ph.read_field_header(d_last, d_fid, d_ft)) {
              if (d_fid == 1 && d_ft == 5) dp_num_values = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 2 && d_ft == 5) dp_encoding = static_cast<int32_t>(ph.read_zigzag());
              else ph.skip_value(d_ft);
            }
          } else if (ph_fid == 8 && ph_ft == 12) {
            int16_t d_last = 0, d_fid = 0;
            uint8_t d_ft = 0;
            while (ph.read_field_header(d_last, d_fid, d_ft)) {
              if (d_fid == 1 && d_ft == 5) dp_num_values = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 4 && d_ft == 5) dp_encoding = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 5 && d_ft == 5) dp2_def_len = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 6 && d_ft == 5) dp2_rep_len = static_cast<int32_t>(ph.read_zigzag());
              else if (d_fid == 7 && (d_ft == 1 || d_ft == 2)) dp2_is_compressed = (d_ft == 1);
              else ph.skip_value(d_ft);
            }
          } else {
            ph.skip_value(ph_ft);
          }
        }
        if (!ph.ok() || compressed_size < 0 || uncompressed_size < 0) {
          ret = OB_INVALID_ARGUMENT;
          return ret;
        }
        size_t payload_start = page_pos + ph.pos();
        if (payload_start + static_cast<size_t>(compressed_size) > file_len) {
          ret = OB_INVALID_ARGUMENT;
          return ret;
        }
        page_pos = payload_start + static_cast<size_t>(compressed_size);

        const uint8_t *payload = raw + payload_start;
        std::vector<uint32_t> def_levels;
        std::vector<uint8_t> decompressed_vals;
        const uint8_t *val_ptr = NULL;
        size_t val_len = 0;

        if (page_type == 3) { // DATA_PAGE_V2
          if (dp2_rep_len + dp2_def_len > compressed_size) {
            ret = OB_INVALID_ARGUMENT;
            return ret;
          }
          const uint8_t *def_ptr = payload + dp2_rep_len;
          if (leaf_cols[j].repetition_type == 1) { // OPTIONAL
            if (!decode_rle_bit_packed(def_ptr, static_cast<size_t>(dp2_def_len), 1,
                                       static_cast<size_t>(dp_num_values), def_levels)) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
          } else {
            def_levels.assign(static_cast<size_t>(dp_num_values), 1);
          }
          const uint8_t *comp_vals = payload + dp2_rep_len + dp2_def_len;
          size_t comp_vals_len = static_cast<size_t>(compressed_size - dp2_rep_len - dp2_def_len);
          if (cmeta.codec == 1 && dp2_is_compressed && comp_vals_len > 0) {
            if (!snappy_decompress(comp_vals, comp_vals_len, decompressed_vals)) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
            val_ptr = decompressed_vals.data();
            val_len = decompressed_vals.size();
          } else if (cmeta.codec == 0 || !dp2_is_compressed) {
            val_ptr = comp_vals;
            val_len = comp_vals_len;
          } else {
            ret = OB_NOT_SUPPORTED;
            return ret;
          }
        } else if (page_type == 0 || page_type == 2) { // DATA_PAGE v1 or DICTIONARY_PAGE
          if (cmeta.codec == 1 && compressed_size > 0) {
            if (!snappy_decompress(payload, static_cast<size_t>(compressed_size), decompressed_vals)) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
            val_ptr = decompressed_vals.data();
            val_len = decompressed_vals.size();
          } else if (cmeta.codec == 0) {
            val_ptr = payload;
            val_len = static_cast<size_t>(compressed_size);
          } else {
            ret = OB_NOT_SUPPORTED;
            return ret;
          }
          if (page_type == 0) {
            if (leaf_cols[j].repetition_type == 1) {
              if (val_len < 4) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              uint32_t def_byte_len = static_cast<uint32_t>(val_ptr[0])
                                    | (static_cast<uint32_t>(val_ptr[1]) << 8)
                                    | (static_cast<uint32_t>(val_ptr[2]) << 16)
                                    | (static_cast<uint32_t>(val_ptr[3]) << 24);
              if (4 + def_byte_len > val_len) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              if (!decode_rle_bit_packed(val_ptr + 4, def_byte_len, 1,
                                         static_cast<size_t>(dp_num_values), def_levels)) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              val_ptr += 4 + def_byte_len;
              val_len -= 4 + def_byte_len;
            } else {
              def_levels.assign(static_cast<size_t>(dp_num_values), 1);
            }
          }
        } else {
          continue;
        }

        auto read_plain_val = [&](size_t &vpos, std::string &out_str) -> bool {
          int32_t ptype = leaf_cols[j].type;
          if (ptype == 1) { // INT32
            if (vpos + 4 > val_len) return false;
            int32_t v = static_cast<int32_t>(static_cast<uint32_t>(val_ptr[vpos])
                      | (static_cast<uint32_t>(val_ptr[vpos + 1]) << 8)
                      | (static_cast<uint32_t>(val_ptr[vpos + 2]) << 16)
                      | (static_cast<uint32_t>(val_ptr[vpos + 3]) << 24));
            vpos += 4;
            out_str = std::to_string(v);
            return true;
          } else if (ptype == 2) { // INT64
            if (vpos + 8 > val_len) return false;
            uint64_t u = 0;
            for (int b = 0; b < 8; ++b) {
              u |= static_cast<uint64_t>(val_ptr[vpos + b]) << (8 * b);
            }
            vpos += 8;
            out_str = std::to_string(static_cast<int64_t>(u));
            return true;
          } else if (ptype == 5) { // DOUBLE
            if (vpos + 8 > val_len) return false;
            double d = 0.0;
            std::memcpy(&d, val_ptr + vpos, 8);
            vpos += 8;
            char dbuf[64];
            std::snprintf(dbuf, sizeof(dbuf), "%.15g", d);
            out_str = dbuf;
            return true;
          } else if (ptype == 6) { // BYTE_ARRAY
            if (vpos + 4 > val_len) return false;
            uint32_t blen = static_cast<uint32_t>(val_ptr[vpos])
                          | (static_cast<uint32_t>(val_ptr[vpos + 1]) << 8)
                          | (static_cast<uint32_t>(val_ptr[vpos + 2]) << 16)
                          | (static_cast<uint32_t>(val_ptr[vpos + 3]) << 24);
            vpos += 4;
            if (vpos + blen > val_len) return false;
            out_str.assign(reinterpret_cast<const char *>(val_ptr + vpos), blen);
            vpos += blen;
            return true;
          }
          return false;
        };

        if (page_type == 2) { // DICTIONARY_PAGE
          size_t vpos = 0;
          dict_vals.clear();
          for (int32_t idx = 0; idx < dp_num_values; ++idx) {
            std::string s;
            if (!read_plain_val(vpos, s)) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
            dict_vals.push_back(s);
          }
          continue;
        }

        size_t non_null_cnt = 0;
        for (size_t idx = 0; idx < def_levels.size(); ++idx) {
          if (def_levels[idx] != 0) {
            ++non_null_cnt;
          }
        }

        std::vector<uint32_t> dict_indices;
        if (dp_encoding == 2 || dp_encoding == 8) {
          if (non_null_cnt > 0) {
            if (val_len < 1) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
            uint8_t bw = val_ptr[0];
            if (!decode_rle_bit_packed(val_ptr + 1, val_len - 1, bw, non_null_cnt, dict_indices)) {
              ret = OB_INVALID_ARGUMENT;
              return ret;
            }
          }
        }

        size_t vpos = 0;
        size_t nn_idx = 0;
        for (int32_t idx = 0; idx < dp_num_values && row_cursor < table.rows.size(); ++idx, ++row_cursor) {
          if (def_levels[idx] == 0) {
            table.rows[row_cursor][j].is_null = true;
            table.cols[j].has_null = true;
          } else {
            table.rows[row_cursor][j].is_null = false;
            if (dp_encoding == 2 || dp_encoding == 8) {
              uint32_t didx = dict_indices[nn_idx++];
              if (didx >= dict_vals.size()) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              table.rows[row_cursor][j].val = dict_vals[didx];
            } else if (leaf_cols[j].type == 0) {
              size_t byte_i = nn_idx / 8;
              size_t bit_i = nn_idx % 8;
              if (byte_i >= val_len) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              bool b = ((val_ptr[byte_i] >> bit_i) & 1) != 0;
              table.rows[row_cursor][j].val = b ? "true" : "false";
              ++nn_idx;
            } else {
              if (!read_plain_val(vpos, table.rows[row_cursor][j].val)) {
                ret = OB_INVALID_ARGUMENT;
                return ret;
              }
              ++nn_idx;
            }
          }
        }
        values_decoded += dp_num_values;
      }
    }
  }
  return ret;
}

static int read_glob_table(const std::string &pattern,
                           const std::vector<NamedOption> &opts,
                           FileTable &table)
{
  int ret = OB_SUCCESS;
  if (!opts.empty()) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported option in glob");
    return ret;
  }
  FileColumn col;
  col.name = "file";
  col.type = COL_VARCHAR;
  col.has_null = false;
  table.cols.push_back(col);

  std::vector<std::string> matched;
#ifdef _WIN32
  std::string norm_pat = pattern;
  for (size_t i = 0; i < norm_pat.size(); ++i) {
    if (norm_pat[i] == '\\') norm_pat[i] = '/';
  }
  std::string dir_prefix;
  size_t last_slash = norm_pat.rfind('/');
  if (last_slash != std::string::npos) {
    dir_prefix = norm_pat.substr(0, last_slash + 1);
  }
  struct _finddata_t fdata;
  intptr_t h = _findfirst(pattern.c_str(), &fdata);
  if (h != -1) {
    do {
      if (0 != std::strcmp(fdata.name, ".") && 0 != std::strcmp(fdata.name, "..")) {
        matched.push_back(dir_prefix + fdata.name);
      }
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
  }
#else
  glob_t gres;
  std::memset(&gres, 0, sizeof(gres));
  int grc = ::glob(pattern.c_str(), 0, NULL, &gres);
  if (grc == 0) {
    for (size_t i = 0; i < gres.gl_pathc; ++i) {
      if (NULL != gres.gl_pathv[i]) {
        matched.push_back(gres.gl_pathv[i]);
      }
    }
  }
  ::globfree(&gres);
#endif
  std::sort(matched.begin(), matched.end());
  for (size_t i = 0; i < matched.size(); ++i) {
    std::vector<FileCell> row(1);
    row[0].is_null = false;
    row[0].val = matched[i];
    table.rows.push_back(row);
  }
  if (table.rows.empty()) {
    table.cols[0].has_null = true;
  }
  return ret;
}

static int load_file_table_func(const std::string &func_name_low,
                                const std::string &args_str,
                                FileTable &table)
{
  int ret = OB_SUCCESS;
  std::string path;
  std::vector<NamedOption> opts;
  if (OB_FAIL(parse_table_func_args(args_str, path, opts))) {
    return ret;
  }
  if (func_name_low == "read_csv") {
    return read_csv_file(path, opts, table);
  } else if (func_name_low == "read_json") {
    return read_json_file(path, opts, table);
  } else if (func_name_low == "read_parquet") {
    return read_parquet_file(path, opts, table);
  } else if (func_name_low == "glob") {
    return read_glob_table(path, opts, table);
  }
  return OB_NOT_SUPPORTED;
}

static void build_table_subquery_sql(const FileTable &table, std::string &out_sql)
{
  out_sql.clear();
  out_sql.push_back('(');
  if (table.rows.empty()) {
    out_sql.append("SELECT ");
    for (size_t j = 0; j < table.cols.size(); ++j) {
      if (j > 0) {
        out_sql.append(", ");
      }
      FileCell null_cell;
      null_cell.is_null = true;
      out_sql.append(format_cell_sql_expr(null_cell, table.cols[j].type));
      out_sql.append(" AS ");
      out_sql.append(escape_sql_ident(table.cols[j].name));
    }
    out_sql.append(" FROM DUAL WHERE 1=0");
  } else {
    for (size_t r = 0; r < table.rows.size(); ++r) {
      if (r == 0) {
        out_sql.append("SELECT ");
      } else {
        out_sql.append(" UNION ALL SELECT ");
      }
      for (size_t j = 0; j < table.cols.size(); ++j) {
        if (j > 0) {
          out_sql.append(", ");
        }
        out_sql.append(format_cell_sql_expr(table.rows[r][j], table.cols[j].type));
        if (r == 0) {
          out_sql.append(" AS ");
          out_sql.append(escape_sql_ident(table.cols[j].name));
        }
      }
    }
  }
  out_sql.push_back(')');
}

static void build_describe_sql(const FileTable &table, std::string &out_sql)
{
  out_sql.clear();
  for (size_t j = 0; j < table.cols.size(); ++j) {
    if (j == 0) {
      out_sql.append("SELECT ");
      out_sql.append(escape_sql_string(table.cols[j].name)).append(" AS `column_name`, ");
      out_sql.append(escape_sql_string(col_type_to_describe_name(table.cols[j].type))).append(" AS `column_type`, ");
      out_sql.append(table.cols[j].has_null ? "'YES'" : "'NO'").append(" AS `null`, ");
      out_sql.append("CAST(NULL AS CHAR(64)) AS `key`, ");
      out_sql.append("CAST(NULL AS CHAR(64)) AS `default`, ");
      out_sql.append("CAST(NULL AS CHAR(64)) AS `extra`");
    } else {
      out_sql.append(" UNION ALL SELECT ");
      out_sql.append(escape_sql_string(table.cols[j].name)).append(", ");
      out_sql.append(escape_sql_string(col_type_to_describe_name(table.cols[j].type))).append(", ");
      out_sql.append(table.cols[j].has_null ? "'YES'" : "'NO'").append(", ");
      out_sql.append("CAST(NULL AS CHAR(64)), CAST(NULL AS CHAR(64)), CAST(NULL AS CHAR(64))");
    }
  }
}

static bool is_sql_clause_keyword(const std::string &w_low)
{
  static const char *const kw[] = {
    "where", "group", "having", "order", "limit", "offset", "fetch",
    "union", "intersect", "except", "minus", "join", "inner", "left",
    "right", "full", "cross", "natural", "straight_join", "on", "using",
    "into", "for", "lock", "window", "partition", "procedure"
  };
  for (size_t i = 0; i < sizeof(kw) / sizeof(kw[0]); ++i) {
    if (w_low == kw[i]) {
      return true;
    }
  }
  return false;
}

static bool find_matching_paren(const std::string &s, size_t open_pos, size_t &close_pos)
{
  if (open_pos >= s.size() || s[open_pos] != '(') {
    return false;
  }
  int depth = 0;
  char in_quote = 0;
  for (size_t i = open_pos; i < s.size(); ++i) {
    char c = s[i];
    if (in_quote != 0) {
      if (c == '\\' && i + 1 < s.size()) {
        ++i;
      } else if (c == in_quote) {
        if (i + 1 < s.size() && s[i + 1] == in_quote) {
          ++i;
        } else {
          in_quote = 0;
        }
      }
    } else {
      if (c == '\'' || c == '"' || c == '`') {
        in_quote = c;
      } else if (c == '(') {
        ++depth;
      } else if (c == ')') {
        --depth;
        if (depth == 0) {
          close_pos = i;
          return true;
        }
      }
    }
  }
  return false;
}

static bool is_file_table_func_name(const std::string &w_low)
{
  return w_low == "read_csv" || w_low == "read_json" || w_low == "read_parquet" || w_low == "glob";
}

struct FileFuncMatch
{
  size_t start_pos;
  size_t open_paren_pos;
  size_t close_paren_pos;
  std::string func_name_low;
  std::string args_str;
};

static int find_file_table_func_calls(const std::string &sql,
                                      std::vector<FileFuncMatch> &matches)
{
  int ret = OB_SUCCESS;
  matches.clear();
  char in_quote = 0;
  size_t i = 0;
  while (i < sql.size()) {
    char c = sql[i];
    if (in_quote != 0) {
      if (c == '\\' && in_quote != '`' && i + 1 < sql.size()) {
        i += 2;
      } else if (c == in_quote) {
        if (i + 1 < sql.size() && sql[i + 1] == in_quote) {
          i += 2;
        } else {
          in_quote = 0;
          ++i;
        }
      } else {
        ++i;
      }
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') {
      in_quote = c;
      ++i;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t w_start = i;
      while (i < sql.size() && is_ident_char(sql[i])) {
        ++i;
      }
      std::string word_low = str_tolower(sql.substr(w_start, i - w_start));
      if (is_file_table_func_name(word_low)) {
        size_t j = i;
        while (j < sql.size() && std::isspace(static_cast<unsigned char>(sql[j]))) {
          ++j;
        }
        if (j < sql.size() && sql[j] == '(') {
          size_t close_p = 0;
          if (!find_matching_paren(sql, j, close_p)) {
            ret = OB_ERR_PARSE_SQL;
            LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                           static_cast<int>(sql.size() - w_start), sql.c_str() + w_start, 1);
            return ret;
          }
          FileFuncMatch m;
          m.start_pos = w_start;
          m.open_paren_pos = j;
          m.close_paren_pos = close_p;
          m.func_name_low = word_low;
          m.args_str = sql.substr(j + 1, close_p - j - 1);
          matches.push_back(m);
          i = close_p + 1;
          continue;
        }
      }
      continue;
    }
    ++i;
  }
  return ret;
}

struct CopyOptions
{
  std::string inner_sql;
  std::string file_path;
  bool has_header;
  char delim;
  char quote;
  std::string nullstr;
  CopyOptions() : inner_sql(), file_path(), has_header(false), delim(','), quote('"'), nullstr("") {}
};

static bool starts_with_copy_stmt(const std::string &sql)
{
  size_t i = 0;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  if (i + 4 > sql.size() || 0 != STRNCASECMP(sql.c_str() + i, "copy", 4)) {
    return false;
  }
  i += 4;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  return i < sql.size() && sql[i] == '(';
}

static bool starts_with_describe_select(const std::string &sql)
{
  size_t i = 0;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  size_t w_start = i;
  while (i < sql.size() && std::isalpha(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  std::string w1 = str_tolower(sql.substr(w_start, i - w_start));
  if (w1 != "describe" && w1 != "desc") {
    return false;
  }
  while (i < sql.size() && (std::isspace(static_cast<unsigned char>(sql[i])) || sql[i] == '(')) {
    ++i;
  }
  size_t w2_start = i;
  while (i < sql.size() && std::isalpha(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  std::string w2 = str_tolower(sql.substr(w2_start, i - w2_start));
  return w2 == "select";
}

static int parse_copy_stmt(const std::string &sql, CopyOptions &out_opts)
{
  int ret = OB_SUCCESS;
  size_t i = 0;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  i += 4; // skip COPY
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  size_t close_query_paren = 0;
  if (i >= sql.size() || sql[i] != '(' || !find_matching_paren(sql, i, close_query_paren)) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(sql.size()), sql.c_str(), 1);
    return ret;
  }
  out_opts.inner_sql = str_trim(sql.substr(i + 1, close_query_paren - i - 1));
  i = close_query_paren + 1;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  if (i + 2 > sql.size() || 0 != STRNCASECMP(sql.c_str() + i, "to", 2)
      || (i + 2 < sql.size() && is_ident_char(sql[i + 2]))) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(sql.size() - i), sql.c_str() + i, 1);
    return ret;
  }
  i += 2;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  if (i >= sql.size() || (sql[i] != '\'' && sql[i] != '"')) {
    ret = OB_ERR_PARSE_SQL;
    LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX),
                   static_cast<int>(sql.size() - i), sql.c_str() + i, 1);
    return ret;
  }
  char q = sql[i];
  size_t q_end = i + 1;
  while (q_end < sql.size()) {
    if (sql[q_end] == '\\' && q_end + 1 < sql.size()) {
      q_end += 2;
    } else if (sql[q_end] == q) {
      if (q_end + 1 < sql.size() && sql[q_end + 1] == q) {
        q_end += 2;
      } else {
        break;
      }
    } else {
      ++q_end;
    }
  }
  if (q_end >= sql.size()) {
    ret = OB_ERR_PARSE_SQL;
    return ret;
  }
  bool was_quoted = false;
  (void)unquote_sql_literal(sql.substr(i, q_end - i + 1), out_opts.file_path, was_quoted);
  i = q_end + 1;
  while (i < sql.size() && std::isspace(static_cast<unsigned char>(sql[i]))) {
    ++i;
  }
  if (i < sql.size() && sql[i] == '(') {
    size_t close_opt_paren = 0;
    if (!find_matching_paren(sql, i, close_opt_paren)) {
      ret = OB_ERR_PARSE_SQL;
      return ret;
    }
    std::string opts_raw = sql.substr(i + 1, close_opt_paren - i - 1);
    size_t p_start = 0;
    char in_q = 0;
    std::vector<std::string> opt_items;
    for (size_t k = 0; k < opts_raw.size(); ++k) {
      if (in_q != 0) {
        if (opts_raw[k] == '\\' && k + 1 < opts_raw.size()) ++k;
        else if (opts_raw[k] == in_q) {
          if (k + 1 < opts_raw.size() && opts_raw[k + 1] == in_q) ++k;
          else in_q = 0;
        }
      } else {
        if (opts_raw[k] == '\'' || opts_raw[k] == '"') in_q = opts_raw[k];
        else if (opts_raw[k] == ',') {
          opt_items.push_back(str_trim(opts_raw.substr(p_start, k - p_start)));
          p_start = k + 1;
        }
      }
    }
    std::string tail_item = str_trim(opts_raw.substr(p_start));
    if (!tail_item.empty()) {
      opt_items.push_back(tail_item);
    }
    for (size_t idx = 0; idx < opt_items.size(); ++idx) {
      const std::string &item = opt_items[idx];
      size_t p = 0;
      while (p < item.size() && is_ident_char(item[p])) {
        ++p;
      }
      std::string k_low = str_tolower(item.substr(0, p));
      while (p < item.size() && (std::isspace(static_cast<unsigned char>(item[p])) || item[p] == '=')) {
        ++p;
      }
      std::string v_raw = str_trim(item.substr(p));
      std::string v_unq;
      bool v_q = false;
      (void)unquote_sql_literal(v_raw, v_unq, v_q);
      std::string v_low = str_tolower(v_unq);
      if (k_low == "format") {
        if (v_low != "csv") {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported format in COPY TO");
          return ret;
        }
      } else if (k_low == "header") {
        if (v_low.empty() || v_low == "true" || v_low == "1") {
          out_opts.has_header = true;
        } else if (v_low == "false" || v_low == "0") {
          out_opts.has_header = false;
        } else {
          ret = OB_NOT_SUPPORTED;
          return ret;
        }
      } else if (k_low == "nullstr" || k_low == "null") {
        out_opts.nullstr = v_unq;
      } else if (k_low == "delimiter" || k_low == "delim" || k_low == "sep") {
        if (v_unq.size() != 1) {
          ret = OB_NOT_SUPPORTED;
          return ret;
        }
        out_opts.delim = v_unq[0];
      } else if (k_low == "quote") {
        if (v_unq.size() != 1) {
          ret = OB_NOT_SUPPORTED;
          return ret;
        }
        out_opts.quote = v_unq[0];
      } else {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "unsupported option in COPY TO");
        return ret;
      }
    }
  }
  return ret;
}

static void write_csv_escaped_field(std::FILE *fp,
                                    const std::string &s,
                                    char delim,
                                    char quote,
                                    const std::string &nullstr)
{
  bool need_quote = false;
  if (s == "NULL" || (!nullstr.empty() && s == nullstr)) {
    need_quote = true;
  }
  for (size_t i = 0; i < s.size() && !need_quote; ++i) {
    if (s[i] == delim || s[i] == quote || s[i] == '\n' || s[i] == '\r') {
      need_quote = true;
    }
  }
  if (!need_quote) {
    if (!s.empty()) {
      std::fwrite(s.data(), 1, s.size(), fp);
    }
  } else {
    std::fputc(quote, fp);
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] == quote) {
        std::fputc(quote, fp);
      }
      std::fputc(s[i], fp);
    }
    std::fputc(quote, fp);
  }
}

static int write_result_set_to_csv(ObResultSet &rs,
                                   ObSQLSessionInfo &session,
                                   const CopyOptions &copy_opts)
{
  int ret = OB_SUCCESS;
  std::FILE *fp = std::fopen(copy_opts.file_path.c_str(), "wb");
  if (NULL == fp) {
    ret = OB_INVALID_ARGUMENT;
    LOG_USER_ERROR(OB_INVALID_ARGUMENT, "cannot open output file in COPY TO");
    return ret;
  }
  const ColumnsFieldIArray *fields = rs.get_field_columns();
  if (copy_opts.has_header && NULL != fields) {
    for (int64_t j = 0; j < fields->count(); ++j) {
      if (j > 0) {
        std::fputc(copy_opts.delim, fp);
      }
      const ObString &cname = fields->at(j).cname_;
      std::string cname_str(cname.ptr(), static_cast<size_t>(cname.length()));
      write_csv_escaped_field(fp, cname_str, copy_opts.delim, copy_opts.quote, copy_opts.nullstr);
    }
    std::fputc('\n', fp);
  }
  ObObjPrintParams print_params;
  print_params.tz_info_ = session.get_timezone_info();
  print_params.use_memcpy_ = true;
  print_params.cs_type_ = CS_TYPE_UTF8MB4_GENERAL_CI;

  const ObNewRow *row = NULL;
  char cell_buf[16384];
  while (OB_SUCC(ret) && OB_SUCC(rs.get_next_row(row))) {
    if (OB_ISNULL(row)) {
      ret = OB_ERR_UNEXPECTED;
      break;
    }
    for (int64_t j = 0; j < row->get_count(); ++j) {
      if (j > 0) {
        std::fputc(copy_opts.delim, fp);
      }
      const ObObj &cell = row->get_cell(j);
      if (cell.is_null()) {
        if (!copy_opts.nullstr.empty()) {
          std::fwrite(copy_opts.nullstr.data(), 1, copy_opts.nullstr.size(), fp);
        }
      } else if (cell.is_string_type()) {
        ObString s = cell.get_string();
        std::string val_str(NULL != s.ptr() ? s.ptr() : "", static_cast<size_t>(s.length()));
        write_csv_escaped_field(fp, val_str, copy_opts.delim, copy_opts.quote, copy_opts.nullstr);
      } else {
        int64_t pos = 0;
        if (OB_FAIL(cell.print_plain_str_literal(cell_buf, sizeof(cell_buf), pos, print_params))) {
          break;
        }
        std::string val_str(cell_buf, static_cast<size_t>(pos));
        write_csv_escaped_field(fp, val_str, copy_opts.delim, copy_opts.quote, copy_opts.nullstr);
      }
    }
    std::fputc('\n', fp);
  }
  if (OB_ITER_END == ret) {
    ret = OB_SUCCESS;
  }
  std::fflush(fp);
  std::fclose(fp);
  return ret;
}

static int rewrite_select_file_funcs(const std::string &sql, std::string &rewritten_sql)
{
  int ret = OB_SUCCESS;
  std::vector<FileFuncMatch> matches;
  if (OB_FAIL(find_file_table_func_calls(sql, matches))) {
    return ret;
  }
  rewritten_sql.clear();
  size_t last_idx = 0;
  for (size_t idx = 0; idx < matches.size(); ++idx) {
    const FileFuncMatch &m = matches[idx];
    rewritten_sql.append(sql, last_idx, m.start_pos - last_idx);
    FileTable tbl;
    if (OB_FAIL(load_file_table_func(m.func_name_low, m.args_str, tbl))) {
      return ret;
    }
    std::string subq;
    build_table_subquery_sql(tbl, subq);
    rewritten_sql.append(subq);

    size_t after_pos = m.close_paren_pos + 1;
    while (after_pos < sql.size() && std::isspace(static_cast<unsigned char>(sql[after_pos]))) {
      ++after_pos;
    }
    bool need_auto_alias = false;
    if (after_pos >= sql.size()
        || sql[after_pos] == ';'
        || sql[after_pos] == ','
        || sql[after_pos] == ')') {
      need_auto_alias = true;
    } else if (std::isalpha(static_cast<unsigned char>(sql[after_pos])) || sql[after_pos] == '_') {
      size_t kw_end = after_pos;
      while (kw_end < sql.size() && is_ident_char(sql[kw_end])) {
        ++kw_end;
      }
      std::string next_word = str_tolower(sql.substr(after_pos, kw_end - after_pos));
      if (is_sql_clause_keyword(next_word)) {
        need_auto_alias = true;
      }
    }
    if (need_auto_alias) {
      rewritten_sql.append(" AS `__file_sql_t").append(std::to_string(idx + 1)).append("`");
    }
    last_idx = m.close_paren_pos + 1;
  }
  rewritten_sql.append(sql, last_idx, sql.size() - last_idx);
  return ret;
}

} // namespace file_sql

namespace ai_column
{

using file_sql::str_tolower;
using file_sql::is_ident_char;

static std::string trim_ws(const std::string &s)
{
  return file_sql::str_trim(s);
}

static std::string sql_escape(const std::string &s)
{
  std::string out;
  out.reserve(s.size() + 8);
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\'') {
      out.append("''");
    } else if (c == '\\') {
      out.append("\\\\");
    } else if (c == '\0') {
      out.append("\\0");
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static thread_local bool g_in_ai_column_internal = false;
static std::once_flag g_ai_worker_once;
static std::atomic<bool> g_ai_meta_tables_ready{false};
static std::atomic<int64_t> g_ai_req_counter{1000};
static std::mutex g_in_flight_mu;
static std::set<std::string> g_in_flight_groups;

struct ScopedInternalGuard
{
  bool prev_;
  ScopedInternalGuard() : prev_(g_in_ai_column_internal) { g_in_ai_column_internal = true; }
  ~ScopedInternalGuard() { g_in_ai_column_internal = prev_; }
};

static int64_t next_request_id()
{
  int64_t now_us = ObTimeUtility::current_time();
  int64_t seq = g_ai_req_counter.fetch_add(1, std::memory_order_relaxed);
  return (now_us / 1000LL) * 1000LL + (seq % 1000LL);
}

static std::string unquote_ident(const std::string &s)
{
  std::string t = trim_ws(s);
  if (t.size() >= 2 && ((t.front() == '`' && t.back() == '`')
                        || (t.front() == '\'' && t.back() == '\'')
                        || (t.front() == '"' && t.back() == '"'))) {
    return t.substr(1, t.size() - 2);
  }
  return t;
}

static std::string quote_ident(const std::string &s)
{
  return "`" + unquote_ident(s) + "`";
}

static std::string strip_trailing_semicolon(const std::string &sql)
{
  std::string s = trim_ws(sql);
  while (!s.empty() && s.back() == ';') {
    s.pop_back();
    s = trim_ws(s);
  }
  return s;
}

static void split_qualified_table_name(const std::string &raw,
                                       const std::string &default_db,
                                       std::string &db_out,
                                       std::string &tbl_out)
{
  std::string s = trim_ws(raw);
  bool in_bt = false;
  size_t dot_pos = std::string::npos;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '`') {
      in_bt = !in_bt;
    } else if (!in_bt && s[i] == '.') {
      dot_pos = i;
      break;
    }
  }
  if (dot_pos != std::string::npos) {
    db_out = unquote_ident(s.substr(0, dot_pos));
    tbl_out = unquote_ident(s.substr(dot_pos + 1));
  } else {
    db_out = default_db;
    tbl_out = unquote_ident(s);
  }
}

static std::string obj_to_string(const ObObj &cell)
{
  if (cell.is_null()) {
    return "";
  }
  if (cell.is_string_type()) {
    ObString s = cell.get_string();
    return std::string(NULL != s.ptr() ? s.ptr() : "", static_cast<size_t>(s.length()));
  }
  if (cell.get_type_class() == ObIntTC) {
    return std::to_string(cell.get_int());
  }
  if (cell.get_type_class() == ObUIntTC) {
    return std::to_string(cell.get_uint64());
  }
  char buf[16384];
  int64_t pos = 0;
  ObObjPrintParams print_params;
  print_params.use_memcpy_ = true;
  print_params.cs_type_ = CS_TYPE_UTF8MB4_GENERAL_CI;
  if (OB_SUCCESS == cell.print_plain_str_literal(buf, sizeof(buf), pos, print_params) && pos > 0) {
    return std::string(buf, static_cast<size_t>(pos));
  }
  return "";
}

static std::string obj_to_sql_literal(const ObObj &cell)
{
  if (cell.is_null()) {
    return "NULL";
  }
  if (cell.get_type_class() == ObIntTC) {
    return std::to_string(cell.get_int());
  }
  if (cell.get_type_class() == ObUIntTC) {
    return std::to_string(cell.get_uint64());
  }
  std::string s = obj_to_string(cell);
  return "'" + sql_escape(s) + "'";
}

struct QueryRow
{
  std::vector<ObObj> cells;
  std::vector<std::string> str_vals;
  std::vector<bool> is_nulls;
};

static int exec_session_sql(ObSql *sql_engine,
                            query::ObIPlanCacheAccessService *pc_service,
                            ObSQLSessionInfo &session,
                            ObSqlCtx &base_ctx,
                            const std::string &sql,
                            std::vector<QueryRow> *out_rows = NULL,
                            bool refresh_schema_after = false)
{
  int ret = OB_SUCCESS;
  ScopedInternalGuard guard;
  ObArenaAllocator inner_alloc(ObModIds::OB_SQL_EXECUTOR, OB_MALLOC_NORMAL_BLOCK_SIZE);
  ObResultSet *inner_rs = OB_NEWx(ObResultSet, (&inner_alloc), session, inner_alloc, *pc_service);
  if (OB_ISNULL(inner_rs)) {
    return OB_ALLOCATE_MEMORY_FAILED;
  }
  {
    LinkExecCtxGuard inner_guard(session, inner_rs->get_exec_context());
    inner_rs->get_exec_context().get_sql_exec_ctx().schema_service_ = GCTX.schema_service_;
    ObSqlCtx inner_ctx = base_ctx;
    ObSchemaGetterGuard latest_guard;
    if (OB_NOT_NULL(GCTX.schema_service_)) {
      if (OB_SUCC(GCTX.schema_service_->get_runtime_schema_guard(latest_guard))) {
        inner_ctx.schema_guard_ = &latest_guard;
      }
    }
    ObString inner_sql_str(static_cast<int32_t>(sql.size()), sql.c_str());
    if (OB_FAIL(inner_rs->init())) {
    } else if (OB_FAIL(sql_engine->stmt_query(inner_sql_str, inner_ctx, *inner_rs))) {
    } else if (OB_FAIL(inner_rs->open())) {
      int cli_ret = OB_SUCCESS;
      (void)inner_rs->close(cli_ret);
    } else {
      if (NULL != out_rows && inner_rs->is_with_rows()) {
        const ObNewRow *row = NULL;
        while (OB_SUCC(ret) && OB_SUCC(inner_rs->get_next_row(row))) {
          if (OB_ISNULL(row)) {
            ret = OB_ERR_UNEXPECTED;
            break;
          }
          QueryRow qrow;
          for (int64_t j = 0; j < row->get_count(); ++j) {
            const ObObj &cell = row->get_cell(j);
            qrow.cells.push_back(cell);
            qrow.is_nulls.push_back(cell.is_null());
            qrow.str_vals.push_back(obj_to_string(cell));
          }
          out_rows->push_back(qrow);
        }
        if (OB_ITER_END == ret) {
          ret = OB_SUCCESS;
        }
      }
      int close_ret = inner_rs->close();
      if (OB_SUCC(ret) && OB_FAIL(close_ret)) {
        ret = close_ret;
      }
    }
  }
  inner_rs->~ObResultSet();
  if (refresh_schema_after) {
    session.get_cached_schema_guard_info().reset();
  }
  return ret;
}

static int proxy_exec_write(const std::string &sql)
{
  if (OB_ISNULL(GCTX.sql_proxy_) || !GCTX.sql_proxy_->is_inited()) {
    return OB_NOT_INIT;
  }
  ScopedInternalGuard guard;
  int64_t affected_rows = 0;
  return GCTX.sql_proxy_->write(sql.c_str(), affected_rows);
}

static int proxy_exec_read(const std::string &sql, int64_t num_cols, std::vector<QueryRow> &out_rows)
{
  int ret = OB_SUCCESS;
  out_rows.clear();
  if (OB_ISNULL(GCTX.sql_proxy_) || !GCTX.sql_proxy_->is_inited()) {
    return OB_NOT_INIT;
  }
  ScopedInternalGuard guard;
  SMART_VAR(ObMySQLProxy::MySQLResult, res) {
    sqlclient::ObMySQLResult *result = NULL;
    if (OB_FAIL(GCTX.sql_proxy_->read(res, sql.c_str()))) {
    } else if (OB_ISNULL(result = res.get_result())) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      while (OB_SUCC(ret) && OB_SUCC(result->next())) {
        QueryRow qrow;
        for (int64_t j = 0; j < num_cols; ++j) {
          ObObj cell;
          if (OB_FAIL(result->get_obj(j, cell))) {
            break;
          }
          qrow.cells.push_back(cell);
          qrow.is_nulls.push_back(cell.is_null());
          qrow.str_vals.push_back(obj_to_string(cell));
        }
        if (OB_SUCC(ret)) {
          out_rows.push_back(qrow);
        }
      }
      if (OB_ITER_END == ret) {
        ret = OB_SUCCESS;
      }
    }
  }
  return ret;
}

static int ensure_metadata_tables_exist()
{
  if (g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  const char *create_meta_sql =
      "CREATE TABLE IF NOT EXISTS oceanbase.__all_ai_column ("
      "table_id BIGINT NOT NULL,"
      "column_id BIGINT NOT NULL,"
      "column_name VARCHAR(128) NOT NULL,"
      "state_column_id BIGINT NOT NULL,"
      "request_column_id BIGINT NOT NULL,"
      "result_spec_column_id BIGINT NOT NULL,"
      "sync_mode VARCHAR(32) NOT NULL DEFAULT 'async',"
      "max_retries INT NOT NULL DEFAULT 3,"
      "normalized_expr VARCHAR(2048),"
      "ai_func_name VARCHAR(64),"
      "model_name VARCHAR(256),"
      "arg_expr VARCHAR(2048),"
      "col_type VARCHAR(128),"
      "dep_columns VARCHAR(1024),"
      "PRIMARY KEY (table_id, column_id))";
  const char *create_task_sql =
      "CREATE TABLE IF NOT EXISTS oceanbase.__all_ai_column_task ("
      "table_id BIGINT NOT NULL,"
      "column_id BIGINT NOT NULL,"
      "rowkey_predicate VARCHAR(512) NOT NULL,"
      "request_id BIGINT NOT NULL,"
      "state INT NOT NULL DEFAULT 1,"
      "retry_count INT NOT NULL DEFAULT 0,"
      "eligible_ts DATETIME(6) NOT NULL,"
      "PRIMARY KEY (table_id, column_id, rowkey_predicate))";
  int ret = proxy_exec_write(create_meta_sql);
  if (OB_SUCC(ret)) {
    ret = proxy_exec_write(create_task_sql);
  }
  if (OB_SUCC(ret)) {
    g_ai_meta_tables_ready.store(true, std::memory_order_release);
  }
  return ret;
}

struct AiColumnMeta
{
  uint64_t table_id = 0;
  uint64_t column_id = 0;
  std::string db_name;
  std::string table_name;
  std::string column_name;
  uint64_t state_column_id = 0;
  uint64_t request_column_id = 0;
  uint64_t result_spec_column_id = 0;
  std::string sync_mode = "async";
  int32_t max_retries = 3;
  std::string normalized_expr;
  std::string ai_func_name;
  std::string model_name;
  std::string arg_expr;
  std::string col_type;
  std::vector<std::string> dep_columns;
};

static std::vector<std::string> split_csv(const std::string &s)
{
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t pos = s.find(',', start);
    std::string item = trim_ws(pos == std::string::npos ? s.substr(start) : s.substr(start, pos - start));
    if (!item.empty()) {
      out.push_back(item);
    }
    if (pos == std::string::npos) {
      break;
    }
    start = pos + 1;
  }
  return out;
}

static int load_table_ai_columns(const std::string &db_name,
                                 const std::string &table_name,
                                 std::vector<AiColumnMeta> &out_cols)
{
  out_cols.clear();
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  std::string sql =
      "SELECT a.table_id, a.column_id, a.column_name, a.state_column_id, a.request_column_id, "
      "a.result_spec_column_id, a.sync_mode, a.max_retries, a.normalized_expr, a.ai_func_name, "
      "a.model_name, a.arg_expr, a.col_type, a.dep_columns "
      "FROM oceanbase.__all_ai_column a "
      "JOIN oceanbase.__all_table t ON t.table_id = a.table_id "
      "JOIN oceanbase.__all_database d ON d.database_id = t.database_id "
      "WHERE d.database_name = '" + sql_escape(db_name) + "' "
      "AND t.table_name = '" + sql_escape(table_name) + "' "
      "ORDER BY a.column_id ASC";
  std::vector<QueryRow> rows;
  int ret = proxy_exec_read(sql, 14, rows);
  if (OB_FAIL(ret)) {
    return OB_SUCCESS;
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    const QueryRow &r = rows[i];
    if (r.str_vals.size() < 14) {
      continue;
    }
    AiColumnMeta m;
    m.table_id = static_cast<uint64_t>(std::strtoull(r.str_vals[0].c_str(), NULL, 10));
    m.column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[1].c_str(), NULL, 10));
    m.db_name = db_name;
    m.table_name = table_name;
    m.column_name = r.str_vals[2];
    m.state_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[3].c_str(), NULL, 10));
    m.request_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[4].c_str(), NULL, 10));
    m.result_spec_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[5].c_str(), NULL, 10));
    m.sync_mode = r.str_vals[6];
    m.max_retries = static_cast<int32_t>(std::strtol(r.str_vals[7].c_str(), NULL, 10));
    m.normalized_expr = r.str_vals[8];
    m.ai_func_name = r.str_vals[9];
    m.model_name = r.str_vals[10];
    m.arg_expr = r.str_vals[11];
    m.col_type = r.str_vals[12];
    m.dep_columns = split_csv(r.str_vals[13]);
    out_cols.push_back(m);
  }
  return OB_SUCCESS;
}

static int load_all_ai_columns(std::vector<AiColumnMeta> &out_cols)
{
  out_cols.clear();
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  std::string sql =
      "SELECT a.table_id, a.column_id, d.database_name, t.table_name, a.column_name, "
      "a.state_column_id, a.request_column_id, a.result_spec_column_id, a.sync_mode, "
      "a.max_retries, a.normalized_expr, a.ai_func_name, a.model_name, a.arg_expr, "
      "a.col_type, a.dep_columns "
      "FROM oceanbase.__all_ai_column a "
      "JOIN oceanbase.__all_table t ON t.table_id = a.table_id "
      "JOIN oceanbase.__all_database d ON d.database_id = t.database_id "
      "ORDER BY a.table_id ASC, a.column_id ASC";
  std::vector<QueryRow> rows;
  int ret = proxy_exec_read(sql, 16, rows);
  if (OB_FAIL(ret)) {
    return OB_SUCCESS;
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    const QueryRow &r = rows[i];
    if (r.str_vals.size() < 16) {
      continue;
    }
    AiColumnMeta m;
    m.table_id = static_cast<uint64_t>(std::strtoull(r.str_vals[0].c_str(), NULL, 10));
    m.column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[1].c_str(), NULL, 10));
    m.db_name = r.str_vals[2];
    m.table_name = r.str_vals[3];
    m.column_name = r.str_vals[4];
    m.state_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[5].c_str(), NULL, 10));
    m.request_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[6].c_str(), NULL, 10));
    m.result_spec_column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[7].c_str(), NULL, 10));
    m.sync_mode = r.str_vals[8];
    m.max_retries = static_cast<int32_t>(std::strtol(r.str_vals[9].c_str(), NULL, 10));
    m.normalized_expr = r.str_vals[10];
    m.ai_func_name = r.str_vals[11];
    m.model_name = r.str_vals[12];
    m.arg_expr = r.str_vals[13];
    m.col_type = r.str_vals[14];
    m.dep_columns = split_csv(r.str_vals[15]);
    out_cols.push_back(m);
  }
  return OB_SUCCESS;
}

static int load_table_pk_columns(uint64_t table_id, std::vector<std::string> &pk_cols)
{
  int ret = OB_SUCCESS;
  pk_cols.clear();
  std::string sql =
      "SELECT column_name FROM oceanbase.__all_column "
      "WHERE table_id = " + std::to_string(table_id) + " "
      "AND rowkey_position > 0 AND is_hidden = 0 "
      "ORDER BY rowkey_position ASC";
  std::vector<QueryRow> rows;
  if (OB_SUCC(proxy_exec_read(sql, 1, rows)) && !rows.empty()) {
    for (size_t i = 0; i < rows.size(); ++i) {
      pk_cols.push_back(rows[i].str_vals[0]);
    }
    return OB_SUCCESS;
  }
  std::string fallback_sql =
      "SELECT column_name FROM oceanbase.__all_column "
      "WHERE table_id = " + std::to_string(table_id) + " "
      "AND rowkey_position > 0 "
      "ORDER BY rowkey_position ASC";
  if (OB_SUCC(proxy_exec_read(fallback_sql, 1, rows))) {
    for (size_t i = 0; i < rows.size(); ++i) {
      pk_cols.push_back(rows[i].str_vals[0]);
    }
  }
  return OB_SUCCESS;
}

static std::string build_pk_select_list(const std::vector<std::string> &pk_cols)
{
  std::string out;
  for (size_t i = 0; i < pk_cols.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += quote_ident(pk_cols[i]);
  }
  return out;
}

static std::string build_rowkey_predicate(const std::vector<std::string> &pk_cols,
                                          const QueryRow &row)
{
  std::string pred;
  for (size_t i = 0; i < pk_cols.size() && i < row.cells.size(); ++i) {
    if (i > 0) {
      pred += " AND ";
    }
    pred += quote_ident(pk_cols[i]) + " = " + obj_to_sql_literal(row.cells[i]);
  }
  return pred;
}

static bool is_integer_col_type(const std::string &col_type)
{
  std::string low = str_tolower(trim_ws(col_type));
  return low.find("int") == 0 || low.find("bigint") == 0 || low.find("smallint") == 0
      || low.find("tinyint") == 0 || low.find("mediumint") == 0 || low.find("integer") == 0;
}

static bool is_json_col_type(const std::string &col_type)
{
  std::string low = str_tolower(trim_ws(col_type));
  return low.find("json") == 0;
}

static bool is_vector_col_type(const std::string &col_type, int64_t &dim_out)
{
  dim_out = -1;
  std::string low = str_tolower(trim_ws(col_type));
  if (low.find("vector") != 0) {
    return false;
  }
  size_t lp = low.find('(');
  size_t rp = low.find(')', lp == std::string::npos ? 0 : lp + 1);
  if (lp != std::string::npos && rp != std::string::npos && rp > lp + 1) {
    std::string dim_str = trim_ws(low.substr(lp + 1, rp - lp - 1));
    dim_out = std::strtoll(dim_str.c_str(), NULL, 10);
  }
  return true;
}

static void ai_worker_process_once()
{
  int ret = OB_SUCCESS;
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return;
  }
  std::string task_sql =
      "SELECT t.table_id, t.column_id, t.rowkey_predicate, t.request_id, t.state, t.retry_count "
      "FROM oceanbase.__all_ai_column_task t "
      "WHERE t.eligible_ts <= NOW(6) "
      "ORDER BY t.table_id ASC, t.rowkey_predicate ASC, t.request_id ASC, t.column_id ASC";
  std::vector<QueryRow> task_rows;
  if (OB_FAIL(proxy_exec_read(task_sql, 6, task_rows)) || task_rows.empty()) {
    return;
  }

  std::vector<AiColumnMeta> all_metas;
  load_all_ai_columns(all_metas);
  std::map<std::pair<uint64_t, uint64_t>, AiColumnMeta> meta_map;
  for (size_t i = 0; i < all_metas.size(); ++i) {
    meta_map[std::make_pair(all_metas[i].table_id, all_metas[i].column_id)] = all_metas[i];
  }

  struct TaskItem
  {
    uint64_t table_id = 0;
    uint64_t column_id = 0;
    std::string rowkey_predicate;
    int64_t request_id = 0;
    int32_t state = 1;
    int32_t retry_count = 0;
  };
  struct TaskGroup
  {
    uint64_t table_id = 0;
    std::string rowkey_predicate;
    int64_t request_id = 0;
    std::string group_key;
    std::vector<TaskItem> items;
  };

  std::vector<TaskGroup> groups;
  for (size_t i = 0; i < task_rows.size(); ++i) {
    const QueryRow &r = task_rows[i];
    if (r.str_vals.size() < 6) {
      continue;
    }
    TaskItem item;
    item.table_id = static_cast<uint64_t>(std::strtoull(r.str_vals[0].c_str(), NULL, 10));
    item.column_id = static_cast<uint64_t>(std::strtoull(r.str_vals[1].c_str(), NULL, 10));
    item.rowkey_predicate = r.str_vals[2];
    item.request_id = static_cast<int64_t>(std::strtoll(r.str_vals[3].c_str(), NULL, 10));
    item.state = static_cast<int32_t>(std::strtol(r.str_vals[4].c_str(), NULL, 10));
    item.retry_count = static_cast<int32_t>(std::strtol(r.str_vals[5].c_str(), NULL, 10));

    std::string gkey = std::to_string(item.table_id) + "|" + item.rowkey_predicate + "|" + std::to_string(item.request_id);
    if (groups.empty() || groups.back().group_key != gkey) {
      TaskGroup g;
      g.table_id = item.table_id;
      g.rowkey_predicate = item.rowkey_predicate;
      g.request_id = item.request_id;
      g.group_key = gkey;
      groups.push_back(g);
    }
    groups.back().items.push_back(item);
  }

  for (size_t gi = 0; gi < groups.size(); ++gi) {
    TaskGroup &grp = groups[gi];
    {
      std::lock_guard<std::mutex> lk(g_in_flight_mu);
      if (g_in_flight_groups.count(grp.group_key) > 0) {
        continue;
      }
      g_in_flight_groups.insert(grp.group_key);
    }

    bool group_stale = false;
    bool group_failed = false;
    std::string db_name;
    std::string table_name;
    std::vector<std::pair<AiColumnMeta, std::string>> computed_cols;

    for (size_t ii = 0; ii < grp.items.size(); ++ii) {
      const TaskItem &item = grp.items[ii];
      auto it = meta_map.find(std::make_pair(item.table_id, item.column_id));
      if (it == meta_map.end()) {
        group_stale = true;
        break;
      }
      const AiColumnMeta &meta = it->second;
      db_name = meta.db_name;
      table_name = meta.table_name;
      std::string req_col = "__ai_col_req_" + meta.column_name;
      std::string eval_sql =
          "SELECT CAST((" + meta.normalized_expr + ") AS CHAR(4000)) "
          "FROM " + quote_ident(db_name) + "." + quote_ident(table_name) + " "
          "WHERE (" + item.rowkey_predicate + ") "
          "AND " + quote_ident(req_col) + " = " + std::to_string(item.request_id);
      std::vector<QueryRow> eval_rows;
      int eval_ret = proxy_exec_read(eval_sql, 1, eval_rows);
      if (OB_FAIL(eval_ret)) {
        group_failed = true;
      } else if (eval_rows.empty()) {
        group_stale = true;
        break;
      } else if (eval_rows[0].is_nulls[0] || eval_rows[0].str_vals[0].empty()) {
        group_failed = true;
      } else {
        computed_cols.push_back(std::make_pair(meta, eval_rows[0].str_vals[0]));
      }

      if (group_failed) {
        std::string state_col = "__ai_col_state_" + meta.column_name;
        if (item.retry_count < meta.max_retries) {
          std::string upd_task_sql =
              "UPDATE oceanbase.__all_ai_column_task "
              "SET state = 3, retry_count = retry_count + 1, "
              "eligible_ts = DATE_ADD(NOW(6), INTERVAL 600000 MICROSECOND) "
              "WHERE table_id = " + std::to_string(item.table_id) + " "
              "AND column_id = " + std::to_string(item.column_id) + " "
              "AND rowkey_predicate = '" + sql_escape(item.rowkey_predicate) + "' "
              "AND request_id = " + std::to_string(item.request_id);
          (void)proxy_exec_write(upd_task_sql);
          std::string upd_row_sql =
              "UPDATE " + quote_ident(db_name) + "." + quote_ident(table_name) + " "
              "SET " + quote_ident(state_col) + " = 'RETRY' "
              "WHERE (" + item.rowkey_predicate + ") "
              "AND " + quote_ident(req_col) + " = " + std::to_string(item.request_id);
          (void)proxy_exec_write(upd_row_sql);
        } else {
          std::string upd_row_sql =
              "UPDATE " + quote_ident(db_name) + "." + quote_ident(table_name) + " "
              "SET " + quote_ident(state_col) + " = 'FAILED' "
              "WHERE (" + item.rowkey_predicate + ") "
              "AND " + quote_ident(req_col) + " = " + std::to_string(item.request_id);
          (void)proxy_exec_write(upd_row_sql);
          std::string del_task_sql =
              "DELETE FROM oceanbase.__all_ai_column_task "
              "WHERE table_id = " + std::to_string(item.table_id) + " "
              "AND column_id = " + std::to_string(item.column_id) + " "
              "AND rowkey_predicate = '" + sql_escape(item.rowkey_predicate) + "' "
              "AND request_id = " + std::to_string(item.request_id);
          (void)proxy_exec_write(del_task_sql);
        }
        break;
      }
    }

    if (group_stale) {
      std::string del_stale_sql =
          "DELETE FROM oceanbase.__all_ai_column_task "
          "WHERE table_id = " + std::to_string(grp.table_id) + " "
          "AND rowkey_predicate = '" + sql_escape(grp.rowkey_predicate) + "' "
          "AND request_id = " + std::to_string(grp.request_id);
      (void)proxy_exec_write(del_stale_sql);
    } else if (!group_failed && !computed_cols.empty()) {
      std::string set_clause;
      std::string req_guard_clause;
      for (size_t ci = 0; ci < computed_cols.size(); ++ci) {
        const AiColumnMeta &meta = computed_cols[ci].first;
        const std::string &val = computed_cols[ci].second;
        std::string state_col = "__ai_col_state_" + meta.column_name;
        std::string req_col = "__ai_col_req_" + meta.column_name;
        if (ci > 0) {
          set_clause += ", ";
        }
        if (is_integer_col_type(meta.col_type)) {
          set_clause += quote_ident(meta.column_name) + " = CAST('" + sql_escape(trim_ws(val)) + "' AS SIGNED)";
        } else if (is_json_col_type(meta.col_type)) {
          set_clause += quote_ident(meta.column_name) + " = CAST('" + sql_escape(val) + "' AS JSON)";
        } else {
          set_clause += quote_ident(meta.column_name) + " = '" + sql_escape(val) + "'";
        }
        set_clause += ", " + quote_ident(state_col) + " = 'READY'";
        req_guard_clause += " AND " + quote_ident(req_col) + " = " + std::to_string(grp.request_id);
      }
      std::string upd_sql =
          "UPDATE " + quote_ident(db_name) + "." + quote_ident(table_name) + " "
          "SET " + set_clause + " "
          "WHERE (" + grp.rowkey_predicate + ")" + req_guard_clause;
      (void)proxy_exec_write(upd_sql);

      std::string del_sql =
          "DELETE FROM oceanbase.__all_ai_column_task "
          "WHERE table_id = " + std::to_string(grp.table_id) + " "
          "AND rowkey_predicate = '" + sql_escape(grp.rowkey_predicate) + "' "
          "AND request_id = " + std::to_string(grp.request_id);
      (void)proxy_exec_write(del_sql);
    }

    {
      std::lock_guard<std::mutex> lk(g_in_flight_mu);
      g_in_flight_groups.erase(grp.group_key);
    }
    break;
  }
}

static void ai_worker_loop()
{
  while (true) {
    int ret = OB_SUCCESS;
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    if (OB_ISNULL(GCTX.sql_proxy_) || !GCTX.sql_proxy_->is_inited()) {
      continue;
    }
    SERVER_MODULE_SCOPE {
      lib::ContextParam param;
      param.set_mem_attr("AIColWorker", ObCtxIds::DEFAULT_CTX_ID)
        .set_properties(lib::USE_TL_PAGE_OPTIONAL)
        .set_page_size(OB_MALLOC_BIG_BLOCK_SIZE);
      CREATE_WITH_TEMP_CONTEXT(param) {
        THIS_WORKER.set_timeout_ts(ObTimeUtility::current_time() + 60LL * 1000000LL);
        ai_worker_process_once();
      }
    }
  }
}

static void ensure_ai_worker_started()
{
  if (OB_NOT_NULL(GCTX.sql_proxy_) && GCTX.sql_proxy_->is_inited()) {
    std::call_once(g_ai_worker_once, []() {
      std::thread(ai_worker_loop).detach();
      std::thread(ai_worker_loop).detach();
    });
  }
}

static std::vector<std::string> split_top_level_comma(const std::string &s)
{
  std::vector<std::string> items;
  int depth = 0;
  char quote = 0;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (quote != 0) {
      if (c == '\\' && quote != '`' && i + 1 < s.size()) {
        ++i;
      } else if (c == quote) {
        if (i + 1 < s.size() && s[i + 1] == quote) {
          ++i;
        } else {
          quote = 0;
        }
      }
    } else {
      if (c == '\'' || c == '"' || c == '`') {
        quote = c;
      } else if (c == '(') {
        ++depth;
      } else if (c == ')') {
        if (depth > 0) {
          --depth;
        }
      } else if (c == ',' && depth == 0) {
        items.push_back(trim_ws(s.substr(start, i - start)));
        start = i + 1;
      }
    }
  }
  std::string tail = trim_ws(s.substr(start));
  if (!tail.empty()) {
    items.push_back(tail);
  }
  return items;
}

static bool find_matching_paren(const std::string &s, size_t open_pos, size_t &close_pos)
{
  if (open_pos >= s.size() || s[open_pos] != '(') {
    return false;
  }
  int depth = 0;
  char quote = 0;
  for (size_t i = open_pos; i < s.size(); ++i) {
    char c = s[i];
    if (quote != 0) {
      if (c == '\\' && quote != '`' && i + 1 < s.size()) {
        ++i;
      } else if (c == quote) {
        if (i + 1 < s.size() && s[i + 1] == quote) {
          ++i;
        } else {
          quote = 0;
        }
      }
    } else {
      if (c == '\'' || c == '"' || c == '`') {
        quote = c;
      } else if (c == '(') {
        ++depth;
      } else if (c == ')') {
        --depth;
        if (depth == 0) {
          close_pos = i;
          return true;
        }
      }
    }
  }
  return false;
}

static size_t find_top_level_keyword_pair(const std::string &s,
                                          const std::string &kw1_low,
                                          const std::string &kw2_low,
                                          size_t *after_kw2_pos = NULL)
{
  int depth = 0;
  char quote = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (quote != 0) {
      if (c == '\\' && quote != '`' && i + 1 < s.size()) {
        ++i;
      } else if (c == quote) {
        if (i + 1 < s.size() && s[i + 1] == quote) {
          ++i;
        } else {
          quote = 0;
        }
      }
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') {
      quote = c;
      continue;
    }
    if (c == '(') {
      ++depth;
      continue;
    }
    if (c == ')') {
      if (depth > 0) {
        --depth;
      }
      continue;
    }
    if (depth == 0 && (std::isalpha(static_cast<unsigned char>(c)) || c == '_')) {
      size_t w1_start = i;
      while (i < s.size() && is_ident_char(s[i])) {
        ++i;
      }
      std::string w1 = str_tolower(s.substr(w1_start, i - w1_start));
      if (w1 == kw1_low) {
        size_t j = i;
        while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) {
          ++j;
        }
        size_t w2_start = j;
        while (j < s.size() && is_ident_char(s[j])) {
          ++j;
        }
        std::string w2 = str_tolower(s.substr(w2_start, j - w2_start));
        if (w2 == kw2_low) {
          if (NULL != after_kw2_pos) {
            *after_kw2_pos = j;
          }
          return w1_start;
        }
      }
      --i;
    }
  }
  return std::string::npos;
}

static size_t find_top_level_keyword(const std::string &s,
                                     const std::string &kw_low,
                                     size_t start_from = 0,
                                     size_t *after_kw_pos = NULL)
{
  int depth = 0;
  char quote = 0;
  for (size_t i = start_from; i < s.size(); ++i) {
    char c = s[i];
    if (quote != 0) {
      if (c == '\\' && quote != '`' && i + 1 < s.size()) {
        ++i;
      } else if (c == quote) {
        if (i + 1 < s.size() && s[i + 1] == quote) {
          ++i;
        } else {
          quote = 0;
        }
      }
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') {
      quote = c;
      continue;
    }
    if (c == '(') {
      ++depth;
      continue;
    }
    if (c == ')') {
      if (depth > 0) {
        --depth;
      }
      continue;
    }
    if (depth == 0 && (std::isalpha(static_cast<unsigned char>(c)) || c == '_')) {
      size_t w_start = i;
      while (i < s.size() && is_ident_char(s[i])) {
        ++i;
      }
      std::string w = str_tolower(s.substr(w_start, i - w_start));
      if (w == kw_low) {
        if (NULL != after_kw_pos) {
          *after_kw_pos = i;
        }
        return w_start;
      }
      --i;
    }
  }
  return std::string::npos;
}

static void extract_referenced_columns(const std::string &expr,
                                       std::vector<std::string> &cols_out)
{
  cols_out.clear();
  std::set<std::string> seen;
  for (size_t i = 0; i < expr.size(); ++i) {
    char c = expr[i];
    if (c == '\'' || c == '"') {
      char q = c;
      ++i;
      while (i < expr.size()) {
        if (expr[i] == '\\' && i + 1 < expr.size()) {
          i += 2;
        } else if (expr[i] == q) {
          if (i + 1 < expr.size() && expr[i + 1] == q) {
            i += 2;
          } else {
            break;
          }
        } else {
          ++i;
        }
      }
    } else if (c == '`') {
      size_t start = i + 1;
      ++i;
      while (i < expr.size() && expr[i] != '`') {
        ++i;
      }
      std::string ident = expr.substr(start, i - start);
      std::string low = str_tolower(ident);
      if (!ident.empty() && seen.find(low) == seen.end()) {
        seen.insert(low);
        cols_out.push_back(ident);
      }
    } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = i;
      while (i < expr.size() && is_ident_char(expr[i])) {
        ++i;
      }
      std::string ident = expr.substr(start, i - start);
      std::string low = str_tolower(ident);
      size_t next_non_ws = i;
      while (next_non_ws < expr.size() && std::isspace(static_cast<unsigned char>(expr[next_non_ws]))) {
        ++next_non_ws;
      }
      bool is_func_call = (next_non_ws < expr.size() && expr[next_non_ws] == '(');
      bool is_keyword = (low == "null" || low == "and" || low == "or" || low == "not"
                         || low == "as" || low == "true" || low == "false"
                         || low == "char" || low == "signed" || low == "unsigned"
                         || low == "json" || low == "case" || low == "when"
                         || low == "then" || low == "else" || low == "end");
      if (!is_func_call && !is_keyword && seen.find(low) == seen.end()) {
        seen.insert(low);
        cols_out.push_back(ident);
      }
      --i;
    }
  }
}

struct ParsedAiColumnDef
{
  std::string column_name;
  std::string col_type;
  std::string ai_func_name;
  std::string model_name;
  std::string arg_expr;
  std::string normalized_expr;
  std::string sync_mode = "async";
  int32_t max_retries = 3;
  std::vector<std::string> dep_columns;
};

static int parse_ai_column_def(const std::string &col_def_str,
                               const std::set<std::string> &regular_cols_low,
                               const std::set<std::string> &ai_cols_low,
                               ParsedAiColumnDef &out_def)
{
  int ret = OB_SUCCESS;
  size_t after_ai_col = 0;
  size_t ai_col_pos = find_top_level_keyword_pair(col_def_str, "ai", "column", &after_ai_col);
  if (ai_col_pos == std::string::npos) {
    return OB_INVALID_ARGUMENT;
  }
  std::string prefix = trim_ws(col_def_str.substr(0, ai_col_pos));
  size_t idx = 0;
  std::string col_name;
  if (!prefix.empty() && prefix[0] == '`') {
    size_t end_bt = prefix.find('`', 1);
    if (end_bt == std::string::npos) {
      return OB_INVALID_ARGUMENT;
    }
    col_name = prefix.substr(1, end_bt - 1);
    idx = end_bt + 1;
  } else {
    while (idx < prefix.size() && !std::isspace(static_cast<unsigned char>(prefix[idx]))) {
      ++idx;
    }
    col_name = prefix.substr(0, idx);
  }
  std::string col_type = trim_ws(prefix.substr(idx));
  if (col_name.empty() || col_type.empty()) {
    return OB_INVALID_ARGUMENT;
  }
  out_def.column_name = col_name;
  out_def.col_type = col_type;

  size_t expr_open = after_ai_col;
  while (expr_open < col_def_str.size() && std::isspace(static_cast<unsigned char>(col_def_str[expr_open]))) {
    ++expr_open;
  }
  size_t expr_close = 0;
  if (!find_matching_paren(col_def_str, expr_open, expr_close)) {
    return OB_INVALID_ARGUMENT;
  }
  std::string inner_expr = trim_ws(col_def_str.substr(expr_open + 1, expr_close - expr_open - 1));

  size_t fn_open = inner_expr.find('(');
  size_t fn_close = 0;
  if (fn_open == std::string::npos || !find_matching_paren(inner_expr, fn_open, fn_close)) {
    return OB_NOT_SUPPORTED;
  }
  std::string fn_name = str_tolower(trim_ws(inner_expr.substr(0, fn_open)));
  if (fn_name != "ai_complete" && fn_name != "ai_embed") {
    return OB_NOT_SUPPORTED;
  }
  out_def.ai_func_name = (fn_name == "ai_complete") ? "AI_COMPLETE" : "AI_EMBED";

  std::string fn_args_str = inner_expr.substr(fn_open + 1, fn_close - fn_open - 1);
  std::vector<std::string> fn_args = split_top_level_comma(fn_args_str);
  if (fn_args.size() < 2) {
    return OB_INVALID_ARGUMENT;
  }
  out_def.model_name = unquote_ident(fn_args[0]);
  out_def.arg_expr = fn_args[1];

  int64_t col_vec_dim = -1;
  bool is_vec_col = is_vector_col_type(col_type, col_vec_dim);
  if (fn_name == "ai_embed" && fn_args.size() >= 3) {
    int64_t arg_dim = std::strtoll(trim_ws(fn_args[2]).c_str(), NULL, 10);
    if (is_vec_col && col_vec_dim > 0 && arg_dim > 0 && col_vec_dim != arg_dim) {
      ret = OB_ERR_INVALID_VECTOR_DIM;
      LOG_USER_ERROR(OB_ERR_INVALID_VECTOR_DIM,
                     static_cast<uint32_t>(col_vec_dim),
                     static_cast<uint32_t>(arg_dim));
      return ret;
    }
  }

  std::vector<std::string> ref_cols;
  extract_referenced_columns(out_def.arg_expr, ref_cols);
  for (size_t i = 0; i < ref_cols.size(); ++i) {
    std::string rlow = str_tolower(ref_cols[i]);
    if (ai_cols_low.count(rlow) > 0 || rlow == str_tolower(col_name)) {
      return OB_NOT_SUPPORTED;
    }
    if (!regular_cols_low.empty() && regular_cols_low.count(rlow) == 0) {
      return OB_ERR_BAD_FIELD_ERROR;
    }
  }
  out_def.dep_columns = ref_cols;

  size_t after_with = 0;
  size_t with_pos = find_top_level_keyword(col_def_str, "with", expr_close + 1, &after_with);
  if (with_pos != std::string::npos) {
    size_t w_open = after_with;
    while (w_open < col_def_str.size() && std::isspace(static_cast<unsigned char>(col_def_str[w_open]))) {
      ++w_open;
    }
    size_t w_close = 0;
    if (!find_matching_paren(col_def_str, w_open, w_close)) {
      return OB_INVALID_ARGUMENT;
    }
    std::vector<std::string> kv_items = split_top_level_comma(col_def_str.substr(w_open + 1, w_close - w_open - 1));
    for (size_t i = 0; i < kv_items.size(); ++i) {
      size_t eq = kv_items[i].find('=');
      if (eq == std::string::npos) {
        return OB_INVALID_ARGUMENT;
      }
      std::string k = str_tolower(trim_ws(kv_items[i].substr(0, eq)));
      std::string v = unquote_ident(kv_items[i].substr(eq + 1));
      if (k == "sync_mode") {
        out_def.sync_mode = str_tolower(v);
        if (out_def.sync_mode != "async") {
          return OB_NOT_SUPPORTED;
        }
      } else if (k == "max_retries") {
        out_def.max_retries = static_cast<int32_t>(std::strtol(v.c_str(), NULL, 10));
        if (out_def.max_retries < 0) {
          return OB_INVALID_ARGUMENT;
        }
      }
    }
  }

  if (out_def.ai_func_name == "AI_COMPLETE") {
    if (is_json_col_type(col_type)) {
      out_def.normalized_expr = "AI_COMPLETE('" + sql_escape(out_def.model_name) + "', " + out_def.arg_expr
          + ", '{\"system_prompt\":\"Return exactly one valid JSON value.\"}')";
    } else if (is_integer_col_type(col_type)) {
      out_def.normalized_expr = "AI_COMPLETE('" + sql_escape(out_def.model_name) + "', " + out_def.arg_expr
          + ", '{\"system_prompt\":\"Return a base-10 integer.\"}')";
    } else {
      out_def.normalized_expr = "AI_COMPLETE('" + sql_escape(out_def.model_name) + "', " + out_def.arg_expr + ")";
    }
  } else {
    out_def.normalized_expr = "AI_EMBED('" + sql_escape(out_def.model_name) + "', " + out_def.arg_expr + ")";
  }
  return OB_SUCCESS;
}

static int persist_ai_column_metadata(const std::string &db_name,
                                      const std::string &table_name,
                                      const std::vector<ParsedAiColumnDef> &ai_defs)
{
  int ret = ensure_metadata_tables_exist();
  if (OB_FAIL(ret)) {
    return ret;
  }
  std::string tbl_sql =
      "SELECT t.table_id FROM oceanbase.__all_table t "
      "JOIN oceanbase.__all_database d ON d.database_id = t.database_id "
      "WHERE d.database_name = '" + sql_escape(db_name) + "' "
      "AND t.table_name = '" + sql_escape(table_name) + "'";
  std::vector<QueryRow> tbl_rows;
  if (OB_FAIL(proxy_exec_read(tbl_sql, 1, tbl_rows)) || tbl_rows.empty()) {
    return OB_ERR_UNEXPECTED;
  }
  uint64_t table_id = static_cast<uint64_t>(std::strtoull(tbl_rows[0].str_vals[0].c_str(), NULL, 10));

  std::string col_sql =
      "SELECT column_name, column_id FROM oceanbase.__all_column "
      "WHERE table_id = " + std::to_string(table_id);
  std::vector<QueryRow> col_rows;
  if (OB_FAIL(proxy_exec_read(col_sql, 2, col_rows))) {
    return ret;
  }
  std::map<std::string, uint64_t> col_id_map;
  for (size_t i = 0; i < col_rows.size(); ++i) {
    col_id_map[str_tolower(col_rows[i].str_vals[0])] =
        static_cast<uint64_t>(std::strtoull(col_rows[i].str_vals[1].c_str(), NULL, 10));
  }

  for (size_t i = 0; i < ai_defs.size(); ++i) {
    const ParsedAiColumnDef &d = ai_defs[i];
    uint64_t col_id = col_id_map[str_tolower(d.column_name)];
    uint64_t state_id = col_id_map[str_tolower("__ai_col_state_" + d.column_name)];
    uint64_t req_id = col_id_map[str_tolower("__ai_col_req_" + d.column_name)];
    uint64_t spec_id = col_id_map[str_tolower("__ai_col_spec_" + d.column_name)];
    std::string dep_csv;
    for (size_t j = 0; j < d.dep_columns.size(); ++j) {
      if (j > 0) {
        dep_csv += ",";
      }
      dep_csv += d.dep_columns[j];
    }
    std::string ins_sql =
        "REPLACE INTO oceanbase.__all_ai_column "
        "(table_id, column_id, column_name, state_column_id, request_column_id, result_spec_column_id, "
        "sync_mode, max_retries, normalized_expr, ai_func_name, model_name, arg_expr, col_type, dep_columns) "
        "VALUES (" + std::to_string(table_id) + ", " + std::to_string(col_id) + ", '"
        + sql_escape(d.column_name) + "', " + std::to_string(state_id) + ", "
        + std::to_string(req_id) + ", " + std::to_string(spec_id) + ", '"
        + sql_escape(d.sync_mode) + "', " + std::to_string(d.max_retries) + ", '"
        + sql_escape(d.normalized_expr) + "', '" + sql_escape(d.ai_func_name) + "', '"
        + sql_escape(d.model_name) + "', '" + sql_escape(d.arg_expr) + "', '"
        + sql_escape(d.col_type) + "', '" + sql_escape(dep_csv) + "')";
    if (OB_FAIL(proxy_exec_write(ins_sql))) {
      return ret;
    }
  }
  return OB_SUCCESS;
}

static bool is_table_constraint_item(const std::string &item)
{
  size_t i = 0;
  while (i < item.size() && std::isspace(static_cast<unsigned char>(item[i]))) {
    ++i;
  }
  size_t start = i;
  while (i < item.size() && is_ident_char(item[i])) {
    ++i;
  }
  std::string w1 = str_tolower(item.substr(start, i - start));
  if (w1 == "primary" || w1 == "unique" || w1 == "key" || w1 == "index"
      || w1 == "constraint" || w1 == "foreign" || w1 == "check" || w1 == "fulltext" || w1 == "vector") {
    return true;
  }
  return false;
}

static int handle_create_table_ai_column(ObSql *sql_engine,
                                         query::ObIPlanCacheAccessService *pc_service,
                                         ObSQLSessionInfo &session,
                                         ObSqlCtx &context,
                                         const std::string &raw_sql,
                                         const std::string &default_db,
                                         bool &handled)
{
  handled = false;
  if (find_top_level_keyword_pair(raw_sql, "create", "table") != 0
      || raw_sql.find("AI COLUMN") == std::string::npos && str_tolower(raw_sql).find("ai column") == std::string::npos) {
    return OB_SUCCESS;
  }
  size_t open_paren = raw_sql.find('(');
  size_t close_paren = 0;
  if (open_paren == std::string::npos || !find_matching_paren(raw_sql, open_paren, close_paren)) {
    return OB_SUCCESS;
  }

  size_t after_create_table = 0;
  find_top_level_keyword_pair(raw_sql, "create", "table", &after_create_table);
  std::string tbl_part = trim_ws(raw_sql.substr(after_create_table, open_paren - after_create_table));
  size_t after_if_not_exists = 0;
  if (find_top_level_keyword_pair(tbl_part, "if", "not") == 0) {
    size_t after_exists = 0;
    if (find_top_level_keyword(tbl_part, "exists", 0, &after_exists) != std::string::npos) {
      tbl_part = trim_ws(tbl_part.substr(after_exists));
    }
  }
  std::string db_name, table_name;
  split_qualified_table_name(tbl_part, default_db, db_name, table_name);

  std::string body = raw_sql.substr(open_paren + 1, close_paren - open_paren - 1);
  std::vector<std::string> items = split_top_level_comma(body);
  std::set<std::string> regular_cols_low;
  std::set<std::string> ai_cols_low;

  for (size_t i = 0; i < items.size(); ++i) {
    if (is_table_constraint_item(items[i])) {
      continue;
    }
    if (find_top_level_keyword_pair(items[i], "ai", "column") != std::string::npos) {
      size_t idx = 0;
      std::string cname;
      if (items[i][0] == '`') {
        size_t e = items[i].find('`', 1);
        if (e != std::string::npos) cname = items[i].substr(1, e - 1);
      } else {
        while (idx < items[i].size() && !std::isspace(static_cast<unsigned char>(items[i][idx]))) ++idx;
        cname = items[i].substr(0, idx);
      }
      ai_cols_low.insert(str_tolower(cname));
    } else {
      size_t idx = 0;
      std::string cname;
      if (items[i][0] == '`') {
        size_t e = items[i].find('`', 1);
        if (e != std::string::npos) cname = items[i].substr(1, e - 1);
      } else {
        while (idx < items[i].size() && !std::isspace(static_cast<unsigned char>(items[i][idx]))) ++idx;
        cname = items[i].substr(0, idx);
      }
      regular_cols_low.insert(str_tolower(cname));
    }
  }

  if (ai_cols_low.empty()) {
    return OB_SUCCESS;
  }
  handled = true;

  std::vector<ParsedAiColumnDef> ai_defs;
  std::string new_body;
  for (size_t i = 0; i < items.size(); ++i) {
    if (!new_body.empty()) {
      new_body += ", ";
    }
    if (!is_table_constraint_item(items[i])
        && find_top_level_keyword_pair(items[i], "ai", "column") != std::string::npos) {
      ParsedAiColumnDef def;
      int ret = parse_ai_column_def(items[i], regular_cols_low, ai_cols_low, def);
      if (OB_FAIL(ret)) {
        return ret;
      }
      ai_defs.push_back(def);
      new_body += quote_ident(def.column_name) + " " + def.col_type
          + ", " + quote_ident("__ai_col_state_" + def.column_name) + " varchar(32)"
          + ", " + quote_ident("__ai_col_req_" + def.column_name) + " bigint"
          + ", " + quote_ident("__ai_col_spec_" + def.column_name) + " varchar(256)";
    } else {
      new_body += items[i];
    }
  }

  std::string rewritten_create = raw_sql.substr(0, open_paren + 1) + new_body + raw_sql.substr(close_paren);
  int ret = exec_session_sql(sql_engine, pc_service, session, context, rewritten_create, NULL, true);
  if (OB_SUCC(ret)) {
    ret = persist_ai_column_metadata(db_name, table_name, ai_defs);
  }
  return ret;
}

static std::string rewrite_retry_where_status(const std::string &pred, const std::string &state_col)
{
  std::string out;
  char quote = 0;
  for (size_t i = 0; i < pred.size(); ++i) {
    char c = pred[i];
    if (quote != 0) {
      out.push_back(c);
      if (c == '\\' && quote != '`' && i + 1 < pred.size()) {
        out.push_back(pred[++i]);
      } else if (c == quote) {
        if (i + 1 < pred.size() && pred[i + 1] == quote) {
          out.push_back(pred[++i]);
        } else {
          quote = 0;
        }
      }
      continue;
    }
    if (c == '\'' || c == '"' || c == '`') {
      quote = c;
      out.push_back(c);
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      bool preceded_by_dot = (!out.empty() && out.back() == '.');
      size_t start = i;
      while (i < pred.size() && is_ident_char(pred[i])) {
        ++i;
      }
      std::string word = pred.substr(start, i - start);
      if (!preceded_by_dot && str_tolower(word) == "status") {
        out += quote_ident(state_col);
      } else {
        out += word;
      }
      --i;
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static int handle_alter_table_ai_column(ObSql *sql_engine,
                                        query::ObIPlanCacheAccessService *pc_service,
                                        ObSQLSessionInfo &session,
                                        ObSqlCtx &context,
                                        const std::string &raw_sql,
                                        const std::string &default_db,
                                        bool &handled)
{
  handled = false;
  size_t after_alter_table = 0;
  if (find_top_level_keyword_pair(raw_sql, "alter", "table", &after_alter_table) != 0) {
    return OB_SUCCESS;
  }
  std::string clean_sql = strip_trailing_semicolon(raw_sql);
  size_t i = after_alter_table;
  while (i < clean_sql.size() && std::isspace(static_cast<unsigned char>(clean_sql[i]))) {
    ++i;
  }
  size_t tbl_start = i;
  bool in_bt = false;
  while (i < clean_sql.size()) {
    if (clean_sql[i] == '`') {
      in_bt = !in_bt;
      ++i;
    } else if (!in_bt && std::isspace(static_cast<unsigned char>(clean_sql[i]))) {
      break;
    } else {
      ++i;
    }
  }
  std::string db_name, table_name;
  split_qualified_table_name(clean_sql.substr(tbl_start, i - tbl_start), default_db, db_name, table_name);
  std::string rest = trim_ws(clean_sql.substr(i));

  // Case A: ALTER TABLE ... RETRY AI COLUMN <col> [WHERE <pred>]
  size_t after_retry = 0;
  if (find_top_level_keyword(rest, "retry", 0, &after_retry) == 0) {
    handled = true;
    std::vector<std::string> actions = split_top_level_comma(rest);
    if (actions.size() > 1) {
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "combining AI COLUMN RETRY with another ALTER action");
      return OB_NOT_SUPPORTED;
    }
    size_t after_ai_col = 0;
    if (find_top_level_keyword_pair(rest, "ai", "column", &after_ai_col) == std::string::npos) {
      return OB_INVALID_ARGUMENT;
    }
    size_t col_start = after_ai_col;
    while (col_start < rest.size() && std::isspace(static_cast<unsigned char>(rest[col_start]))) {
      ++col_start;
    }
    size_t after_where = 0;
    size_t where_pos = find_top_level_keyword(rest, "where", col_start, &after_where);
    std::string col_name = unquote_ident(where_pos == std::string::npos
                                         ? rest.substr(col_start)
                                         : rest.substr(col_start, where_pos - col_start));
    std::string where_pred = (where_pos == std::string::npos) ? "" : trim_ws(rest.substr(after_where));

    std::vector<AiColumnMeta> ai_cols;
    load_table_ai_columns(db_name, table_name, ai_cols);
    const AiColumnMeta *target_meta = NULL;
    for (size_t ci = 0; ci < ai_cols.size(); ++ci) {
      if (str_tolower(ai_cols[ci].column_name) == str_tolower(col_name)) {
        target_meta = &ai_cols[ci];
        break;
      }
    }
    if (NULL == target_meta) {
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "retrying a column that is not an AI column");
      return OB_NOT_SUPPORTED;
    }

    std::vector<std::string> pk_cols;
    load_table_pk_columns(target_meta->table_id, pk_cols);
    if (pk_cols.empty()) {
      return OB_ERR_UNEXPECTED;
    }
    std::string state_col = "__ai_col_state_" + target_meta->column_name;
    std::string req_col = "__ai_col_req_" + target_meta->column_name;
    std::string rewritten_where = where_pred.empty() ? "1 = 1" : rewrite_retry_where_status(where_pred, state_col);

    std::string sel_pks_sql =
        "SELECT " + build_pk_select_list(pk_cols) + " FROM "
        + quote_ident(db_name) + "." + quote_ident(table_name)
        + " WHERE " + rewritten_where;
    std::vector<QueryRow> pk_rows;
    int ret = exec_session_sql(sql_engine, pc_service, session, context, sel_pks_sql, &pk_rows);
    if (OB_FAIL(ret) || pk_rows.empty()) {
      return ret;
    }
    for (size_t ri = 0; ri < pk_rows.size(); ++ri) {
      std::string rpred = build_rowkey_predicate(pk_cols, pk_rows[ri]);
      int64_t req_id = next_request_id();
      std::string upd_sql =
          "UPDATE " + quote_ident(db_name) + "." + quote_ident(table_name) + " SET "
          + quote_ident(target_meta->column_name) + " = NULL, "
          + quote_ident(state_col) + " = 'PENDING', "
          + quote_ident(req_col) + " = " + std::to_string(req_id)
          + " WHERE " + rpred;
      if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, upd_sql))) {
        return ret;
      }
      std::string task_sql =
          "REPLACE INTO oceanbase.__all_ai_column_task "
          "(table_id, column_id, rowkey_predicate, request_id, state, retry_count, eligible_ts) VALUES ("
          + std::to_string(target_meta->table_id) + ", "
          + std::to_string(target_meta->column_id) + ", '"
          + sql_escape(rpred) + "', " + std::to_string(req_id) + ", 1, 0, NOW(6))";
      if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, task_sql))) {
        return ret;
      }
    }
    return OB_SUCCESS;
  }

  // Case B: ALTER TABLE ... ADD [COLUMN] <col> <type> AI COLUMN (...)
  if (find_top_level_keyword_pair(rest, "ai", "column") != std::string::npos) {
    handled = true;
    size_t after_add = 0;
    if (find_top_level_keyword(rest, "add", 0, &after_add) != 0) {
      return OB_NOT_SUPPORTED;
    }
    std::string col_def = trim_ws(rest.substr(after_add));
    size_t after_col_kw = 0;
    if (find_top_level_keyword(col_def, "column", 0, &after_col_kw) == 0) {
      col_def = trim_ws(col_def.substr(after_col_kw));
    }

    std::vector<AiColumnMeta> existing_ai;
    load_table_ai_columns(db_name, table_name, existing_ai);
    std::set<std::string> ai_cols_low;
    for (size_t ci = 0; ci < existing_ai.size(); ++ci) {
      ai_cols_low.insert(str_tolower(existing_ai[ci].column_name));
    }
    std::set<std::string> regular_cols_low;
    std::string col_sql =
        "SELECT c.column_name FROM oceanbase.__all_column c "
        "JOIN oceanbase.__all_table t ON t.table_id = c.table_id "
        "JOIN oceanbase.__all_database d ON d.database_id = t.database_id "
        "WHERE d.database_name = '" + sql_escape(db_name) + "' "
        "AND t.table_name = '" + sql_escape(table_name) + "' AND c.is_hidden = 0";
    int ret = OB_SUCCESS;
    std::vector<QueryRow> col_rows;
    if (OB_SUCC(proxy_exec_read(col_sql, 1, col_rows))) {
      for (size_t ci = 0; ci < col_rows.size(); ++ci) {
        std::string clow = str_tolower(col_rows[ci].str_vals[0]);
        if (ai_cols_low.count(clow) == 0) {
          regular_cols_low.insert(clow);
        }
      }
    }

    ParsedAiColumnDef def;
    ret = parse_ai_column_def(col_def, regular_cols_low, ai_cols_low, def);
    if (OB_FAIL(ret)) {
      return ret;
    }
    std::string rewritten_alter =
        "ALTER TABLE " + quote_ident(db_name) + "." + quote_ident(table_name)
        + " ADD COLUMN " + quote_ident(def.column_name) + " " + def.col_type
        + ", ADD COLUMN " + quote_ident("__ai_col_state_" + def.column_name) + " varchar(32)"
        + ", ADD COLUMN " + quote_ident("__ai_col_req_" + def.column_name) + " bigint"
        + ", ADD COLUMN " + quote_ident("__ai_col_spec_" + def.column_name) + " varchar(256)";
    if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, rewritten_alter, NULL, true))) {
      return ret;
    }
    std::vector<ParsedAiColumnDef> defs{def};
    return persist_ai_column_metadata(db_name, table_name, defs);
  }

  return OB_SUCCESS;
}

static int handle_drop_table_ai_column(const std::string &raw_sql,
                                       const std::string &default_db)
{
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  size_t after_drop_db = 0;
  if (find_top_level_keyword_pair(raw_sql, "drop", "database", &after_drop_db) == 0) {
    std::string rest = trim_ws(strip_trailing_semicolon(raw_sql).substr(after_drop_db));
    size_t after_exists = 0;
    if (find_top_level_keyword_pair(rest, "if", "exists", &after_exists) == 0) {
      rest = trim_ws(rest.substr(after_exists));
    }
    std::string db_name = unquote_ident(rest);
    std::vector<AiColumnMeta> all_cols;
    load_all_ai_columns(all_cols);
    for (size_t i = 0; i < all_cols.size(); ++i) {
      if (str_tolower(all_cols[i].db_name) == str_tolower(db_name)) {
        uint64_t tid = all_cols[i].table_id;
        (void)proxy_exec_write("DELETE FROM oceanbase.__all_ai_column WHERE table_id = " + std::to_string(tid));
        (void)proxy_exec_write("DELETE FROM oceanbase.__all_ai_column_task WHERE table_id = " + std::to_string(tid));
      }
    }
    return OB_SUCCESS;
  }
  size_t after_drop_table = 0;
  if (find_top_level_keyword_pair(raw_sql, "drop", "table", &after_drop_table) != 0) {
    return OB_SUCCESS;
  }
  std::string rest = trim_ws(strip_trailing_semicolon(raw_sql).substr(after_drop_table));
  size_t after_exists = 0;
  if (find_top_level_keyword_pair(rest, "if", "exists", &after_exists) == 0) {
    rest = trim_ws(rest.substr(after_exists));
  }
  std::vector<std::string> tbls = split_top_level_comma(rest);
  for (size_t i = 0; i < tbls.size(); ++i) {
    std::string db_name, table_name;
    split_qualified_table_name(tbls[i], default_db, db_name, table_name);
    std::vector<AiColumnMeta> cols;
    load_table_ai_columns(db_name, table_name, cols);
    if (!cols.empty()) {
      uint64_t tid = cols[0].table_id;
      (void)proxy_exec_write("DELETE FROM oceanbase.__all_ai_column WHERE table_id = " + std::to_string(tid));
      (void)proxy_exec_write("DELETE FROM oceanbase.__all_ai_column_task WHERE table_id = " + std::to_string(tid));
    }
  }
  return OB_SUCCESS;
}

static bool sql_references_identifier(const std::string &sql, const std::string &ident)
{
  std::string target_low = str_tolower(ident);
  char quote = 0;
  for (size_t i = 0; i < sql.size(); ++i) {
    char c = sql[i];
    if (quote != 0) {
      if (c == '\\' && quote != '`' && i + 1 < sql.size()) {
        ++i;
      } else if (c == quote) {
        if (i + 1 < sql.size() && sql[i + 1] == quote) {
          ++i;
        } else {
          quote = 0;
        }
      }
      continue;
    }
    if (c == '\'' || c == '"') {
      quote = c;
      continue;
    }
    if (c == '`') {
      size_t start = i + 1;
      ++i;
      while (i < sql.size() && sql[i] != '`') {
        ++i;
      }
      if (str_tolower(sql.substr(start, i - start)) == target_low) {
        return true;
      }
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = i;
      while (i < sql.size() && is_ident_char(sql[i])) {
        ++i;
      }
      if (str_tolower(sql.substr(start, i - start)) == target_low) {
        return true;
      }
      --i;
    }
  }
  return false;
}

static int check_strict_read_for_sql(ObSql *sql_engine,
                                     query::ObIPlanCacheAccessService *pc_service,
                                     ObSQLSessionInfo &session,
                                     ObSqlCtx &context,
                                     const std::string &raw_sql,
                                     const std::string &default_db)
{
  int ret = OB_SUCCESS;
  int64_t strict_read = 0;
  if (OB_FAIL(session.get_sys_variable(share::SYS_VAR_AI_COLUMN_STRICT_READ, strict_read)) || 0 == strict_read) {
    return OB_SUCCESS;
  }
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  std::string low_sql = str_tolower(raw_sql);
  if (low_sql.find("__all_ai_column") != std::string::npos) {
    return OB_SUCCESS;
  }
  std::vector<AiColumnMeta> all_metas;
  load_all_ai_columns(all_metas);
  for (size_t i = 0; i < all_metas.size(); ++i) {
    const AiColumnMeta &m = all_metas[i];
    bool table_mentioned = sql_references_identifier(raw_sql, m.table_name)
        && (str_tolower(m.db_name) == str_tolower(default_db)
            || sql_references_identifier(raw_sql, m.db_name));
    if (!table_mentioned) {
      continue;
    }
    if (!sql_references_identifier(raw_sql, m.column_name)) {
      continue;
    }
    std::string state_col = "__ai_col_state_" + m.column_name;
    std::string chk_sql =
        "SELECT COUNT(*) FROM " + quote_ident(m.db_name) + "." + quote_ident(m.table_name)
        + " WHERE " + quote_ident(state_col) + " IS NULL OR " + quote_ident(state_col) + " != 'READY'";
    std::vector<QueryRow> rows;
    int ret = exec_session_sql(sql_engine, pc_service, session, context, chk_sql, &rows);
    if (OB_SUCC(ret) && !rows.empty() && std::strtoll(rows[0].str_vals[0].c_str(), NULL, 10) > 0) {
      LOG_USER_ERROR(OB_AI_COLUMN_UNFINISHED_TASKS, m.table_id, m.column_id);
      return OB_AI_COLUMN_UNFINISHED_TASKS;
    }
  }
  return OB_SUCCESS;
}

static int handle_dml_ai_column(ObSql *sql_engine,
                                query::ObIPlanCacheAccessService *pc_service,
                                ObSQLSessionInfo &session,
                                ObSqlCtx &context,
                                const std::string &raw_sql,
                                const std::string &default_db,
                                bool &handled)
{
  handled = false;
  if (!g_ai_meta_tables_ready.load(std::memory_order_acquire)) {
    return OB_SUCCESS;
  }
  std::string clean_sql = strip_trailing_semicolon(raw_sql);
  size_t pos = 0;
  while (pos < clean_sql.size() && std::isspace(static_cast<unsigned char>(clean_sql[pos]))) {
    ++pos;
  }
  size_t w_start = pos;
  while (pos < clean_sql.size() && is_ident_char(clean_sql[pos])) {
    ++pos;
  }
  std::string first_kw = str_tolower(clean_sql.substr(w_start, pos - w_start));

  if (first_kw == "select") {
    return check_strict_read_for_sql(sql_engine, pc_service, session, context, clean_sql, default_db);
  }

  // Case 1: INSERT INTO <tbl> ...
  if (first_kw == "insert") {
    size_t after_into = 0;
    if (find_top_level_keyword(clean_sql, "into", pos, &after_into) == std::string::npos) {
      return OB_SUCCESS;
    }
    size_t i = after_into;
    while (i < clean_sql.size() && std::isspace(static_cast<unsigned char>(clean_sql[i]))) {
      ++i;
    }
    size_t tbl_start = i;
    bool in_bt = false;
    while (i < clean_sql.size()) {
      if (clean_sql[i] == '`') {
        in_bt = !in_bt;
        ++i;
      } else if (!in_bt && (std::isspace(static_cast<unsigned char>(clean_sql[i])) || clean_sql[i] == '(')) {
        break;
      } else {
        ++i;
      }
    }
    std::string db_name, table_name;
    split_qualified_table_name(clean_sql.substr(tbl_start, i - tbl_start), default_db, db_name, table_name);
    if (str_tolower(db_name) == "oceanbase") {
      return OB_SUCCESS;
    }
    std::vector<AiColumnMeta> ai_cols;
    load_table_ai_columns(db_name, table_name, ai_cols);
    if (ai_cols.empty()) {
      return OB_SUCCESS;
    }
    handled = true;
    int ret = exec_session_sql(sql_engine, pc_service, session, context, clean_sql);
    if (OB_FAIL(ret)) {
      return ret;
    }
    std::vector<std::string> pk_cols;
    load_table_pk_columns(ai_cols[0].table_id, pk_cols);
    if (pk_cols.empty()) {
      return OB_ERR_UNEXPECTED;
    }
    std::string first_state_col = "__ai_col_state_" + ai_cols[0].column_name;
    std::string sel_new_sql =
        "SELECT " + build_pk_select_list(pk_cols) + " FROM "
        + quote_ident(db_name) + "." + quote_ident(table_name)
        + " WHERE " + quote_ident(first_state_col) + " IS NULL";
    std::vector<QueryRow> new_rows;
    if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, sel_new_sql, &new_rows))) {
      return ret;
    }
    for (size_t ri = 0; ri < new_rows.size(); ++ri) {
      std::string rpred = build_rowkey_predicate(pk_cols, new_rows[ri]);
      int64_t req_id = next_request_id();
      std::string set_clause;
      for (size_t ci = 0; ci < ai_cols.size(); ++ci) {
        if (ci > 0) {
          set_clause += ", ";
        }
        const AiColumnMeta &m = ai_cols[ci];
        set_clause += quote_ident(m.column_name) + " = NULL, "
            + quote_ident("__ai_col_state_" + m.column_name) + " = 'PENDING', "
            + quote_ident("__ai_col_req_" + m.column_name) + " = " + std::to_string(req_id) + ", "
            + quote_ident("__ai_col_spec_" + m.column_name) + " = '" + sql_escape(m.col_type) + "'";
      }
      std::string upd_sql =
          "UPDATE " + quote_ident(db_name) + "." + quote_ident(table_name)
          + " SET " + set_clause + " WHERE " + rpred;
      if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, upd_sql))) {
        return ret;
      }
      for (size_t ci = 0; ci < ai_cols.size(); ++ci) {
        const AiColumnMeta &m = ai_cols[ci];
        std::string task_sql =
            "REPLACE INTO oceanbase.__all_ai_column_task "
            "(table_id, column_id, rowkey_predicate, request_id, state, retry_count, eligible_ts) VALUES ("
            + std::to_string(m.table_id) + ", " + std::to_string(m.column_id) + ", '"
            + sql_escape(rpred) + "', " + std::to_string(req_id) + ", 1, 0, NOW(6))";
        if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, task_sql))) {
          return ret;
        }
      }
    }
    return OB_SUCCESS;
  }

  // Case 2: UPDATE <tbl> SET ... [WHERE ...]
  if (first_kw == "update") {
    int ret = check_strict_read_for_sql(sql_engine, pc_service, session, context, clean_sql, default_db);
    if (OB_FAIL(ret)) {
      return ret;
    }
    size_t after_set = 0;
    size_t set_pos = find_top_level_keyword(clean_sql, "set", pos, &after_set);
    if (set_pos == std::string::npos) {
      return OB_SUCCESS;
    }
    std::string tbl_raw = trim_ws(clean_sql.substr(pos, set_pos - pos));
    std::string db_name, table_name;
    split_qualified_table_name(tbl_raw, default_db, db_name, table_name);
    if (str_tolower(db_name) == "oceanbase") {
      return OB_SUCCESS;
    }
    std::vector<AiColumnMeta> ai_cols;
    load_table_ai_columns(db_name, table_name, ai_cols);
    if (ai_cols.empty()) {
      return OB_SUCCESS;
    }
    size_t after_where = 0;
    size_t where_pos = find_top_level_keyword(clean_sql, "where", after_set, &after_where);
    std::string set_body = (where_pos == std::string::npos)
        ? trim_ws(clean_sql.substr(after_set))
        : trim_ws(clean_sql.substr(after_set, where_pos - after_set));
    std::string where_clause = (where_pos == std::string::npos)
        ? ""
        : trim_ws(clean_sql.substr(after_where));

    std::vector<std::string> assigns = split_top_level_comma(set_body);
    std::set<std::string> updated_cols_low;
    for (size_t ai = 0; ai < assigns.size(); ++ai) {
      size_t eq = assigns[ai].find('=');
      if (eq != std::string::npos) {
        std::string lhs = unquote_ident(assigns[ai].substr(0, eq));
        size_t dot = lhs.rfind('.');
        if (dot != std::string::npos) {
          lhs = unquote_ident(lhs.substr(dot + 1));
        }
        updated_cols_low.insert(str_tolower(lhs));
      }
    }

    std::vector<std::string> pk_cols;
    load_table_pk_columns(ai_cols[0].table_id, pk_cols);
    bool pk_modified = false;
    for (size_t pi = 0; pi < pk_cols.size(); ++pi) {
      if (updated_cols_low.count(str_tolower(pk_cols[pi])) > 0) {
        pk_modified = true;
        break;
      }
    }

    std::vector<AiColumnMeta> affected_ai;
    for (size_t ci = 0; ci < ai_cols.size(); ++ci) {
      bool dep_hit = pk_modified;
      for (size_t di = 0; !dep_hit && di < ai_cols[ci].dep_columns.size(); ++di) {
        if (updated_cols_low.count(str_tolower(ai_cols[ci].dep_columns[di])) > 0) {
          dep_hit = true;
        }
      }
      if (dep_hit) {
        affected_ai.push_back(ai_cols[ci]);
      }
    }
    if (affected_ai.empty()) {
      return OB_SUCCESS;
    }

    handled = true;
    if (pk_modified) {
      std::string old_pk_sql =
          "SELECT " + build_pk_select_list(pk_cols) + " FROM "
          + quote_ident(db_name) + "." + quote_ident(table_name)
          + (where_clause.empty() ? "" : (" WHERE " + where_clause));
      std::vector<QueryRow> old_rows;
      if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, old_pk_sql, &old_rows))) {
        return ret;
      }
      for (size_t ri = 0; ri < old_rows.size(); ++ri) {
        std::string old_pred = build_rowkey_predicate(pk_cols, old_rows[ri]);
        std::string del_old_sql =
            "DELETE FROM oceanbase.__all_ai_column_task "
            "WHERE table_id = " + std::to_string(ai_cols[0].table_id)
            + " AND rowkey_predicate = '" + sql_escape(old_pred) + "'";
        if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, del_old_sql))) {
          return ret;
        }
      }
    }

    int64_t req_id = next_request_id();
    std::string extra_set;
    for (size_t ci = 0; ci < affected_ai.size(); ++ci) {
      const AiColumnMeta &m = affected_ai[ci];
      extra_set += ", " + quote_ident(m.column_name) + " = NULL, "
          + quote_ident("__ai_col_state_" + m.column_name) + " = 'PENDING', "
          + quote_ident("__ai_col_req_" + m.column_name) + " = " + std::to_string(req_id) + ", "
          + quote_ident("__ai_col_spec_" + m.column_name) + " = '" + sql_escape(m.col_type) + "'";
    }
    std::string rewritten_upd =
        clean_sql.substr(0, after_set) + " " + set_body + extra_set
        + (where_clause.empty() ? "" : (" WHERE " + where_clause));
    if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, rewritten_upd))) {
      return ret;
    }

    std::string first_req_col = "__ai_col_req_" + affected_ai[0].column_name;
    std::string new_pk_sql =
        "SELECT " + build_pk_select_list(pk_cols) + " FROM "
        + quote_ident(db_name) + "." + quote_ident(table_name)
        + " WHERE " + quote_ident(first_req_col) + " = " + std::to_string(req_id);
    std::vector<QueryRow> new_rows;
    if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, new_pk_sql, &new_rows))) {
      return ret;
    }
    for (size_t ri = 0; ri < new_rows.size(); ++ri) {
      std::string rpred = build_rowkey_predicate(pk_cols, new_rows[ri]);
      for (size_t ci = 0; ci < affected_ai.size(); ++ci) {
        const AiColumnMeta &m = affected_ai[ci];
        std::string task_sql =
            "REPLACE INTO oceanbase.__all_ai_column_task "
            "(table_id, column_id, rowkey_predicate, request_id, state, retry_count, eligible_ts) VALUES ("
            + std::to_string(m.table_id) + ", " + std::to_string(m.column_id) + ", '"
            + sql_escape(rpred) + "', " + std::to_string(req_id) + ", 1, 0, NOW(6))";
        if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, task_sql))) {
          return ret;
        }
      }
    }
    return OB_SUCCESS;
  }

  // Case 3: DELETE FROM <tbl> [WHERE ...]
  if (first_kw == "delete") {
    int ret = check_strict_read_for_sql(sql_engine, pc_service, session, context, clean_sql, default_db);
    if (OB_FAIL(ret)) {
      return ret;
    }
    size_t after_from = 0;
    if (find_top_level_keyword(clean_sql, "from", pos, &after_from) == std::string::npos) {
      return OB_SUCCESS;
    }
    size_t after_where = 0;
    size_t where_pos = find_top_level_keyword(clean_sql, "where", after_from, &after_where);
    std::string tbl_raw = (where_pos == std::string::npos)
        ? trim_ws(clean_sql.substr(after_from))
        : trim_ws(clean_sql.substr(after_from, where_pos - after_from));
    std::string where_clause = (where_pos == std::string::npos)
        ? ""
        : trim_ws(clean_sql.substr(after_where));
    std::string db_name, table_name;
    split_qualified_table_name(tbl_raw, default_db, db_name, table_name);
    if (str_tolower(db_name) == "oceanbase") {
      return OB_SUCCESS;
    }
    std::vector<AiColumnMeta> ai_cols;
    load_table_ai_columns(db_name, table_name, ai_cols);
    if (ai_cols.empty()) {
      return OB_SUCCESS;
    }
    handled = true;
    std::vector<std::string> pk_cols;
    load_table_pk_columns(ai_cols[0].table_id, pk_cols);
    std::string sel_pk_sql =
        "SELECT " + build_pk_select_list(pk_cols) + " FROM "
        + quote_ident(db_name) + "." + quote_ident(table_name)
        + (where_clause.empty() ? "" : (" WHERE " + where_clause));
    std::vector<QueryRow> del_rows;
    if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, sel_pk_sql, &del_rows))) {
      return ret;
    }
    for (size_t ri = 0; ri < del_rows.size(); ++ri) {
      std::string rpred = build_rowkey_predicate(pk_cols, del_rows[ri]);
      std::string del_task_sql =
          "DELETE FROM oceanbase.__all_ai_column_task "
          "WHERE table_id = " + std::to_string(ai_cols[0].table_id)
          + " AND rowkey_predicate = '" + sql_escape(rpred) + "'";
      if (OB_FAIL(ret = exec_session_sql(sql_engine, pc_service, session, context, del_task_sql))) {
        return ret;
      }
    }
    return exec_session_sql(sql_engine, pc_service, session, context, clean_sql);
  }

  return OB_SUCCESS;
}

} // namespace ai_column

OB_INLINE int ObSql::handle_text_query(const ObString &stmt, ObSqlCtx &context, ObResultSet &result)
{
  int ret = OB_SUCCESS;
  //trim the sql first, let 'select c1 from t' and '  select c1 from t' and hit the same plan_cache
  ObString trimed_stmt = const_cast<ObString &>(stmt).trim();
  context.is_prepare_protocol_ = false;
  if (OB_FAIL(init_result_set(context, result))) {
  } else if (trimed_stmt.empty()) {
    ret = OB_ERR_EMPTY_QUERY;
    LOG_USER_ERROR(OB_ERR_EMPTY_QUERY);
    // Empty request, can be categorized as a known error of parser, no need to disconnect
    result.get_exec_context().set_need_disconnect(false);
    //FIXME qianfu NG_TRACE_EXT(set_need_disconnect, OB_ID(need_disconnect), false);
  }

  ObIAllocator &allocator = THIS_WORKER.get_sql_arena_allocator();
  ObSQLSessionInfo &session = result.get_session();
  
  ObExecContext& ectx = result.get_exec_context();
  int get_plan_err = OB_SUCCESS; //used for judge whether add plan to plan cache
  bool use_plan_cache = session.get_local_ob_enable_plan_cache();
  ObPlanCacheCtx *pc_ctx = NULL;
  bool is_begin_commit_stmt = false;
  bool is_file_sql_rewritten = false;

  if (OB_SUCC(ret) && !trimed_stmt.empty()) {
    std::string raw_sql(trimed_stmt.ptr(), static_cast<size_t>(trimed_stmt.length()));
    bool is_copy = file_sql::starts_with_copy_stmt(raw_sql);
    bool is_desc_select = file_sql::starts_with_describe_select(raw_sql);
    std::vector<file_sql::FileFuncMatch> func_matches;
    int find_ret = file_sql::find_file_table_func_calls(raw_sql, func_matches);
    if (is_copy || is_desc_select || OB_FAIL(find_ret) || !func_matches.empty()) {
      ectx.set_need_disconnect(false);
      int64_t enable_file_sql = 0;
      if (OB_FAIL(session.get_sys_variable(share::SYS_VAR_ENABLE_FILE_SQL, enable_file_sql))) {
      } else if (0 == enable_file_sql) {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "File SQL when enable_file_sql is disabled");
      } else if (OB_FAIL(find_ret)) {
        ret = find_ret;
      } else if (is_copy) {
        file_sql::CopyOptions copy_opts;
        if (OB_FAIL(file_sql::parse_copy_stmt(raw_sql, copy_opts))) {
        } else {
          ObArenaAllocator copy_alloc(ObModIds::OB_SQL_EXECUTOR, OB_MALLOC_NORMAL_BLOCK_SIZE);
          ObResultSet *inner_rs = OB_NEWx(ObResultSet, (&copy_alloc), session, copy_alloc, *plan_cache_access_service_);
          if (OB_ISNULL(inner_rs)) {
            ret = OB_ALLOCATE_MEMORY_FAILED;
          } else {
            {
              LinkExecCtxGuard inner_guard(session, inner_rs->get_exec_context());
              ObSqlCtx inner_ctx = context;
              ObString inner_sql_str(static_cast<int32_t>(copy_opts.inner_sql.size()), copy_opts.inner_sql.c_str());
              if (OB_FAIL(inner_rs->init())) {
              } else if (OB_FAIL(this->handle_text_query(inner_sql_str, inner_ctx, *inner_rs))) {
              } else if (OB_FAIL(inner_rs->open())) {
                int cli_ret = OB_SUCCESS;
                (void)inner_rs->close(cli_ret);
              } else {
                ret = file_sql::write_result_set_to_csv(*inner_rs, session, copy_opts);
                int close_ret = inner_rs->close();
                if (OB_SUCC(ret) && OB_FAIL(close_ret)) {
                  ret = close_ret;
                }
              }
            }
            inner_rs->~ObResultSet();
          }
          if (OB_SUCC(ret)) {
            const char *noop_sql = "SET @@session.enable_file_sql = @@session.enable_file_sql";
            if (OB_FAIL(ob_write_string(allocator, ObString::make_string(noop_sql), trimed_stmt))) {
            } else {
              is_file_sql_rewritten = true;
            }
          }
        }
      } else if (is_desc_select && !func_matches.empty()) {
        file_sql::FileTable tbl;
        std::string desc_sql;
        if (OB_FAIL(file_sql::load_file_table_func(func_matches[0].func_name_low, func_matches[0].args_str, tbl))) {
        } else {
          file_sql::build_describe_sql(tbl, desc_sql);
          if (OB_FAIL(ob_write_string(allocator, ObString(static_cast<int32_t>(desc_sql.size()), desc_sql.c_str()), trimed_stmt))) {
          } else {
            is_file_sql_rewritten = true;
          }
        }
      } else if (!func_matches.empty()) {
        std::string rewritten_sql;
        if (OB_FAIL(file_sql::rewrite_select_file_funcs(raw_sql, rewritten_sql))) {
        } else if (OB_FAIL(ob_write_string(allocator, ObString(static_cast<int32_t>(rewritten_sql.size()), rewritten_sql.c_str()), trimed_stmt))) {
        } else {
          is_file_sql_rewritten = true;
        }
      }
    }

    if (OB_SUCC(ret) && !is_file_sql_rewritten && !ai_column::g_in_ai_column_internal && !session.is_inner()) {
      ai_column::ensure_ai_worker_started();
      ObString db_obstr = session.get_database_name();
      std::string default_db(db_obstr.ptr() != NULL ? db_obstr.ptr() : "",
                             db_obstr.ptr() != NULL ? static_cast<size_t>(db_obstr.length()) : 0);
      bool ai_handled = false;
      if (OB_FAIL(ai_column::handle_create_table_ai_column(
              this, plan_cache_access_service_, session, context, raw_sql, default_db, ai_handled))) {
        ectx.set_need_disconnect(false);
      } else if (!ai_handled && OB_FAIL(ai_column::handle_alter_table_ai_column(
              this, plan_cache_access_service_, session, context, raw_sql, default_db, ai_handled))) {
        ectx.set_need_disconnect(false);
      } else if (!ai_handled && OB_FAIL(ai_column::handle_drop_table_ai_column(raw_sql, default_db))) {
        ectx.set_need_disconnect(false);
      } else if (!ai_handled && OB_FAIL(ai_column::handle_dml_ai_column(
              this, plan_cache_access_service_, session, context, raw_sql, default_db, ai_handled))) {
        ectx.set_need_disconnect(false);
      }
      if (OB_SUCC(ret) && ai_handled) {
        ectx.set_need_disconnect(false);
        const char *noop_sql = "SET @@session.ai_column_strict_read = @@session.ai_column_strict_read";
        if (OB_FAIL(ob_write_string(allocator, ObString::make_string(noop_sql), trimed_stmt))) {
        } else {
          is_file_sql_rewritten = true;
        }
      }
    }
  }

  if (OB_FAIL(ret)) {
    // do nothing
  } else if (NULL == (pc_ctx = static_cast<ObPlanCacheCtx *>
                               (allocator.alloc(sizeof(ObPlanCacheCtx))))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    context.cur_sql_ = trimed_stmt;
    pc_ctx = new (pc_ctx) ObPlanCacheCtx(trimed_stmt,
                                         PC_TEXT_MODE,
                                         allocator,
                                         context,
                                         ectx);
    if (trimed_stmt.length() == 6) {
      // Whether it is a COMMIT statement
      is_begin_commit_stmt = (0 == STRNCASECMP(trimed_stmt.ptr(), "commit", 6)
                              && !context.is_batch_params_execute());
    } else if (trimed_stmt.length() == 5) {
      is_begin_commit_stmt = (0 == STRNCASECMP(trimed_stmt.ptr(), "begin", 5)
                              && !context.is_batch_params_execute());
    }
    if (is_begin_commit_stmt) {
      // Record current statement is begin/commit statement, used for performance optimization
      pc_ctx->set_begin_commit_stmt();
    }
    uint64_t database_id = OB_INVALID_ID;

    if (OB_FAIL(session.get_database_id(database_id))) {
    } else if (FALSE_IT(context.plan_key_.db_id_ =
                                  (database_id == OB_INVALID_ID) ?
                                      OB_MOCK_DEFAULT_DATABASE_ID:
                                      database_id)) {
      // do nothing
    } else if (is_file_sql_rewritten) {
      // Skip plan cache lookup and force direct compilation for rewritten File SQL queries
    } else if (!use_plan_cache) {
      if (context.multi_stmt_item_.is_batched_multi_stmt()) {
        ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
        LOG_WARN("batched multi_stmt needs rollback");
      }
      // If it is a begin/commit statement, do not get the plan from the plan cache
    } else if (!is_begin_commit_stmt
        && OB_FAIL(pc_get_plan_and_fill_result(*pc_ctx, result, get_plan_err,
                                               ectx.get_need_disconnect_for_update()))) {
    }
  }

  if (OB_SUCC(ret) && !result.get_is_from_plan_cache()) { // did not get plan from plan cache, take the long path to generate plan
    if (OB_FAIL(SMART_CALL_LARGE(handle_physical_plan(trimed_stmt, context, result, *pc_ctx, get_plan_err)))) {
    }
  }

  if ((NULL != pc_ctx) && !(pc_ctx->sql_ctx_.is_sensitive_)) {
    // if sql context contains sensitive data, can not flush sql info to trace.log
  }

  // set auto-increment related param into physical plan ctx
  // if get plan from plan cache, reset its auto-increment variable here
  // do not set variable for hidden primary key; its default value is 1
  if (OB_SUCC(ret)) {
    if (!context.is_text_ps_mode_
        && OB_FAIL(after_get_plan(*pc_ctx, session, result.get_physical_plan(),
                                  result.get_is_from_plan_cache()))) {
    }
  }
  if (NULL != pc_ctx) {
    pc_ctx->~ObPlanCacheCtx();
  }
  return ret;
}

int ObSql::generate_stmt(ParseResult &parse_result,
                         ObPlanCacheCtx *pc_ctx,
                         ObSqlCtx &context,
                         ObIAllocator &allocator,
                         ObResultSet &result,
                         ObStmt *&stmt,
                         ParseResult *outline_parse_result)
{
  int ret = OB_SUCCESS;
  uint64_t session_id = 0;
  ObResolverParams resolver_ctx;
  ObPhysicalPlanCtx *plan_ctx = NULL;
  ObSchemaChecker *schema_checker = NULL;

  int64_t last_mem_usage = allocator.total();
  int64_t resolver_mem_usage = 0;
  if (OB_FAIL(sanity_check(context))) {
  } else {
    if (result.get_session().get_session_type() != ObSQLSessionInfo::INNER_SESSION) {
      session_id = result.get_session().get_sessid_for_table();
    } else {
      session_id = OB_INVALID_ID; // internal session, not affected by table_schema->session_id visibility, can see tables during query and table creation process
    }
    schema_checker = OB_NEWx(ObSchemaChecker, (&allocator));
    if (OB_UNLIKELY(NULL == schema_checker)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_ERROR("Failed to malloc ObSchemaChecker", K(ret));
    } else if (NULL == context.schema_guard_) {
      ret = OB_ERR_UNEXPECTED;
    }
  }

  if (OB_SUCC(ret)) {
    resolver_ctx.allocator_  = &allocator;
    resolver_ctx.schema_checker_ = schema_checker;
    resolver_ctx.secondary_namespace_ = context.secondary_namespace_;
    resolver_ctx.session_info_ = context.session_info_;
    bind_resolver_runtime_services(resolver_ctx);
    resolver_ctx.expr_factory_ = result.get_exec_context().get_expr_factory();
    resolver_ctx.stmt_factory_ = result.get_exec_context().get_stmt_factory();
    resolver_ctx.cur_sql_ = context.cur_sql_;
    resolver_ctx.is_restore_ = context.is_restore_;
    resolver_ctx.is_ddl_from_primary_ = context.is_ddl_from_primary_;
    resolver_ctx.is_cursor_ = context.is_cursor_;
    resolver_ctx.is_batch_stmt_ = context.is_batch_params_execute();
    resolver_ctx.batch_stmt_num_ = context.get_batch_params_count();
    resolver_ctx.need_check_col_dup_
      = !(context.is_prepare_protocol_ && parse_result.question_mark_ctx_.by_ordinal_);
    resolver_ctx.external_param_info_.by_name_
        = parse_result.question_mark_ctx_.by_name_ || NULL != context.secondary_namespace_; //static sql in PL must be by name
    resolver_ctx.outline_parse_result_ = outline_parse_result;
    resolver_ctx.is_execute_call_stmt_ = context.is_execute_call_stmt_;
    if (NULL != pc_ctx) {
      resolver_ctx.select_item_param_infos_ = &pc_ctx->select_item_param_infos_;
    }
    plan_ctx = result.get_exec_context().get_physical_plan_ctx();
    if (OB_ISNULL(plan_ctx) || OB_ISNULL(result.get_exec_context().get_stmt_factory())
        || OB_ISNULL(result.get_exec_context().get_expr_factory())) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_ISNULL(resolver_ctx.query_ctx_ =
        result.get_exec_context().get_stmt_factory()->get_query_ctx())) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_ERROR("allocate query context failed", K(ret));
    } else {
      resolver_ctx.query_ctx_->sql_schema_guard_.set_schema_guard(context.schema_guard_);
      resolver_ctx.expr_factory_->set_query_ctx(resolver_ctx.query_ctx_);
      if (OB_FAIL(resolver_ctx.schema_checker_->init(resolver_ctx.query_ctx_->sql_schema_guard_, session_id))) {
      }
    }
  }
  if (OB_SUCC(ret)) {
    if (NULL != resolver_ctx.secondary_namespace_
        && NULL !=  resolver_ctx.secondary_namespace_->get_external_ns()) {
      resolver_ctx.package_guard_ =
        &resolver_ctx.secondary_namespace_->get_external_ns()->get_resolve_ctx().package_guard_;
    }
  }
  if (OB_SUCC(ret)) {
    resolver_ctx.is_prepare_protocol_ = context.is_prepare_protocol_;
    resolver_ctx.is_prepare_stage_ = context.is_prepare_stage_;
    resolver_ctx.is_mock_prepare_ = context.is_mock_prepare_;
    resolver_ctx.is_dynamic_sql_ = context.is_dynamic_sql_;
    resolver_ctx.statement_id_ = context.statement_id_;
    resolver_ctx.param_list_ = &plan_ctx->get_param_store();
    resolver_ctx.sql_proxy_ = GCTX.sql_proxy_;
  }

  if (OB_FAIL(ret)) {
  } else if (stmt::T_ANONYMOUS_BLOCK == context.stmt_type_ && context.is_prepare_protocol_ && !context.is_prepare_stage_) {
    // anonymous + ps will not do parser in execute stage, therefore should not check parser_result
    //do nothing...
  } else if (OB_ISNULL(parse_result.result_tree_)
        || OB_ISNULL(parse_result.result_tree_->children_)
        || OB_ISNULL(parse_result.result_tree_->children_[0])) {
      ret = OB_INVALID_ARGUMENT;
      SQL_LOG(WARN, "invalid args",
              KP(parse_result.result_tree_),
              KP(parse_result.result_tree_->children_),
              KP(parse_result.result_tree_->children_[0]));
  }

  if (OB_SUCC(ret)) {
      // set # of question marks
      if (context.is_prepare_protocol_ && !context.is_prepare_stage_) {
        resolver_ctx.query_ctx_->set_questionmark_count(plan_ctx->get_param_store().count());
        LOG_DEBUG("question mark size is ", K(plan_ctx->get_param_store()));
      } else {
        resolver_ctx.query_ctx_->set_questionmark_count(static_cast<int64_t>(parse_result.question_mark_ctx_.count_));
      }

      HEAP_VAR(ObResolver, resolver, resolver_ctx) {
      NG_TRACE(resolve_begin);
      if (OB_FAIL(plan_ctx->build_subschema_ctx_by_param_store(context.schema_guard_))) {
      } else if (stmt::T_ANONYMOUS_BLOCK == context.stmt_type_
          && context.is_prepare_protocol_
          && !context.is_prepare_stage_
          && !context.is_mock_prepare_) {
        ParseNode tmp_node;
        tmp_node.type_ = T_SP_ANONYMOUS_BLOCK;
        ret = resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT, tmp_node, stmt);
      } else {
        ret = resolver.resolve(ObResolver::IS_NOT_PREPARED_STMT, *parse_result.result_tree_->children_[0], stmt);
      }

      // set const param constraint after resolving
      context.all_plan_const_param_constraints_ = &(resolver_ctx.query_ctx_->all_plan_const_param_constraints_);
      context.all_possible_const_param_constraints_ = &(resolver_ctx.query_ctx_->all_possible_const_param_constraints_);
      context.all_equal_param_constraints_ = &(resolver_ctx.query_ctx_->all_equal_param_constraints_);
      context.all_pre_calc_constraints_ = &(resolver_ctx.query_ctx_->all_pre_calc_constraints_);
      context.all_expr_constraints_ = &(resolver_ctx.query_ctx_->all_expr_constraints_);
      context.all_priv_constraints_ = &(resolver_ctx.query_ctx_->all_priv_constraints_);
      context.need_match_all_params_ = resolver_ctx.query_ctx_->need_match_all_params_;
      context.all_local_session_vars_ = &(resolver_ctx.query_ctx_->all_local_session_vars_);
      context.cur_stmt_ = stmt;
      NG_TRACE(resolve_end);
      if (OB_SUCC(ret)) {
        if (NULL != pc_ctx && NULL != pc_ctx->exec_ctx_.get_physical_plan_ctx()) {
          if (OB_FAIL(pc_ctx->exec_ctx_.get_physical_plan_ctx()->init_datum_param_store())) {
          }
        }
      }
      //add ref obj schema version to PL and ps info
      if (OB_SUCC(ret)) {
        if (OB_FAIL(result.get_ref_objects().assign(resolver_ctx.query_ctx_->global_dependency_tables_))) {
        }
      }
      if (OB_FAIL(ret)) {
        /* for audit */
        if (NULL != stmt) {
          result.set_stmt_type(stmt->get_stmt_type());
        }
        SQL_LOG(WARN, "failed to resolve", K(ret));
      } else {
        // process stmt
        if (NULL != stmt && NULL != resolver_ctx.query_ctx_) {
          SQL_LOG(DEBUG, "SET STMT PARAM COUNT", K(resolver.get_params().prepare_param_count_), K(&resolver_ctx));
          // secondary_namespace_ is not empty, indicating that it is the prepare stage of sql in PL
          // PL dynamic SQL needs its external parameters rebuilt after resolving.
          bool need_rebuild = (resolver_ctx.is_dynamic_sql_ &&
          OB_NOT_NULL(result.get_session().get_pl_context()) && resolver_ctx.is_prepare_stage_);
          if (need_rebuild) {
            if (OB_FAIL(result.get_external_retrieve_info().build(*stmt,
                result.get_session(),
                resolver_ctx.secondary_namespace_,
                resolver_ctx.is_dynamic_sql_,
                resolver.get_params().external_param_info_.params_))) {
            } else {
              if (result.get_external_params().empty() && result.get_into_exprs().empty()) {
                if (resolver_ctx.query_ctx_->get_sql_stmt().empty()) {
                  resolver_ctx.query_ctx_->set_sql_stmt(parse_result.input_sql_, parse_result.input_sql_len_);
                }
                result.get_route_sql() = resolver_ctx.query_ctx_->get_sql_stmt();
                resolver_ctx.query_ctx_->set_prepare_param_count(parse_result.question_mark_ctx_.count_);
              } else {
                // For SQL statements within PL, the specific prepare_param_count_
                // can only be determined after resolver completion.
                resolver_ctx.query_ctx_->set_prepare_param_count(resolver.get_params().prepare_param_count_);
              }
            }
          } else {
            if (stmt::T_ANONYMOUS_BLOCK == context.stmt_type_ && context.is_prepare_protocol_) {
              resolver_ctx.query_ctx_->set_sql_stmt(context.cur_sql_);
            } else {
              resolver_ctx.query_ctx_->set_sql_stmt(parse_result.input_sql_, parse_result.input_sql_len_);
            }
            result.get_route_sql() = resolver_ctx.query_ctx_->get_sql_stmt();
            resolver_ctx.query_ctx_->set_prepare_param_count(parse_result.question_mark_ctx_.count_);
          }
          if (OB_SUCC(ret)) {
            result.set_stmt_type(stmt->get_stmt_type()); // used by query retry
            result.set_literal_stmt_type(resolver_ctx.query_ctx_->get_literal_stmt_type()); // used by show warn, show trace
            ObDMLStmt *dml_stmt = NULL;
            ObVariableSetStmt *variable_set_stmt = NULL;
            if ((dml_stmt = dynamic_cast<ObDMLStmt*>(stmt)) != NULL) {
              const ObGlobalHint &global_hint = resolver_ctx.query_ctx_->get_global_hint();
              result.set_is_calc_found_rows(dml_stmt->is_calc_found_rows());
              plan_ctx->set_is_affect_found_row(dml_stmt->is_affect_found_rows());
              context.force_print_trace_ = global_hint.force_trace_log_;
              if (MpQuery == context.exec_type_ && global_hint.log_level_.length() > 0) {
                const ObString &log_level = global_hint.log_level_;
                if (OB_UNLIKELY(OB_SUCCESS != process_thread_log_id_level_map(log_level.ptr(),
                                                                              log_level.length()))) {
                }
              }
            } else if ((variable_set_stmt = dynamic_cast<ObVariableSetStmt*>(stmt)) != NULL) {
              result.set_has_global_variable(variable_set_stmt->has_global_variable());
            }
          }
          if (OB_SUCC(ret)) {
          } else {
          }
        } else {
          ret = OB_ERR_UNEXPECTED;
          SQL_LOG(WARN, "failed to generate stmt", K(ret));
        }
      }
      } // end HEAP_VAR(ObResolver)
  }
  resolver_mem_usage = allocator.total() - last_mem_usage;
  return ret;
}

int ObSql::generate_physical_plan(ParseResult &parse_result,
                                  ObPlanCacheCtx *pc_ctx,
                                  ObSqlCtx &sql_ctx,
                                  ObResultSet &result,
                                  const bool is_begin_commit_stmt,
                                  const PlanCacheMode mode,
                                  ParseResult *outline_parse_result /* null */ )
{
  int ret = OB_SUCCESS;
  bool is_valid = true;
  ObStmt *basic_stmt = NULL;
  ObIAllocator &allocator = result.get_mem_pool();
  ObStmtNeedPrivs stmt_need_privs;
  stmt_need_privs.need_privs_.set_allocator(&allocator);
  if (OB_FAIL(sanity_check(sql_ctx))) {
  } else if (OB_FAIL(generate_stmt(parse_result,
                                   pc_ctx,
                                   sql_ctx,
                                   allocator,
                                   result,
                                   basic_stmt,
                                   outline_parse_result))) {
  } else if (OB_ISNULL(basic_stmt)) {
    ret = OB_ERR_UNEXPECTED;
    // begin/commit statement does not need to check privilege
  } else if (!is_begin_commit_stmt
          && OB_FAIL(ObPrivilegeCheck::check_privilege_new(sql_ctx,
                                                           basic_stmt,
                                                           stmt_need_privs))) {
    LOG_WARN("Failed to check ora privilege info", K(ret), K(*basic_stmt));
  } else if (OB_FAIL(ObPrivilegeCheck::check_password_expired(sql_ctx,
                                                              basic_stmt->get_stmt_type()))) {
  } else if ((sql_ctx.is_batch_params_execute()) &&
             NULL != pc_ctx &&
             OB_FAIL(check_batched_multi_stmt_after_resolver(*pc_ctx,
                                                             *basic_stmt,
                                                             is_valid))) {
  } else if (!is_valid) {
    ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
    LOG_WARN("batched multi_stmt needs rollback", K(ret));
  } else if (basic_stmt->is_dml_stmt()
             || basic_stmt->is_explain_stmt()
            ) {
    if (OB_FAIL(generate_plan(parse_result, pc_ctx, sql_ctx, result, mode, basic_stmt,
                             stmt_need_privs))) {
    }
  } else if (stmt::T_EXECUTE == basic_stmt->get_stmt_type() &&
             stmt::T_CALL_PROCEDURE != static_cast<ObExecuteStmt*>(basic_stmt)->get_prepare_type()) {
    if (OB_FAIL(handle_text_execute(basic_stmt, sql_ctx, result))) {
    }
  } else {
    ObICmd *cmd = dynamic_cast<ObICmd*>(basic_stmt);
    if (OB_UNLIKELY(NULL == cmd)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      result.set_cmd(cmd);
      result.get_session().set_cur_sql_id(sql_ctx.sql_id_);
      if (!is_begin_commit_stmt
        && OB_FAIL(fill_result_set(result, &sql_ctx, mode, *basic_stmt))) {
      }
    }
  }
  return ret;
}

int ObSql::generate_plan(ParseResult &parse_result,
                        ObPlanCacheCtx *pc_ctx,
                        ObSqlCtx &sql_ctx,
                        ObResultSet &result,
                        const PlanCacheMode mode,
                        ObStmt *basic_stmt,
                        ObStmtNeedPrivs &stmt_need_privs)
{
  int ret = OB_SUCCESS;
  uint64_t aggregate_setting = 0;
  ObPhysicalPlanCtx *pctx = result.get_exec_context().get_physical_plan_ctx();
  ObPlanCache *plan_cache = plan_cache_;
  bool allow_audit = false;
  if (OB_ISNULL(pctx) || OB_ISNULL(basic_stmt) ||
      OB_ISNULL(result.get_exec_context().get_expr_factory()) ||
      OB_ISNULL(plan_cache)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(result.get_exec_context().get_stmt_factory()->get_query_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(fill_result_set(result, &sql_ctx, mode, *basic_stmt))) {
  } else if (OB_FAIL(sql_ctx.session_info_->get_sys_variable(
                      share::SYS_VAR__AGGREGATION_OPTIMIZATION_SETTINGS,
                      aggregate_setting))) {
  } else {
    ObDMLStmt *stmt = static_cast<ObDMLStmt*>(basic_stmt);
    stmt->get_query_ctx()->root_stmt_ = stmt;
    const ObGlobalHint &global_hint = stmt->get_query_ctx()->get_global_hint();
    HEAP_VAR(ObOptimizerContext, optctx, sql_ctx.session_info_,
                              &result.get_exec_context(),
                              &result.get_exec_context().get_stmt_factory()->get_query_ctx()->sql_schema_guard_,
                              opt_stat_mgr_,
                              result.get_mem_pool(),
                              &pctx->get_param_store(),
                              self_addr_,
                              global_hint,
                              *result.get_exec_context().get_expr_factory(),
                              stmt,
                              result.is_ps_protocol(),
                              result.get_exec_context().get_stmt_factory()->get_query_ctx()) {
    optctx.set_aggregation_optimization_settings(aggregate_setting);
    pctx->set_field_array(result.get_field_columns());
    pctx->set_is_ps_protocol(result.is_ps_protocol());
    bool is_restore = false;
    ObOptimizer optimizer(optctx);

    ObLogPlan *logical_plan = NULL;
    ObPhysicalPlan *phy_plan = NULL;
    ObCacheObjGuard& guard = result.get_cache_obj_guard();
    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(ObCacheObjectFactory::alloc(*plan_cache, guard,
                                                    ObLibCacheNameSpace::NS_CRSR))) {
    } else if (FALSE_IT(phy_plan = static_cast<ObPhysicalPlan*>(guard.get_cache_obj()))) {
      // do nothing
    } else if (OB_UNLIKELY(NULL == phy_plan)) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_ERROR("Failed to alloc physical plan from tc factory", K(ret));
    } else {
      if (NULL != pc_ctx) {
        pc_ctx->should_add_plan_ = true;
      }
    }

    if (OB_SUCC(ret)) {
      phy_plan->set_fetch_cur_time(stmt->get_fetch_cur_time());
      phy_plan->set_stmt_type(stmt->get_stmt_type());
      phy_plan->set_literal_stmt_type(stmt->get_query_ctx()->get_literal_stmt_type());
      if (phy_plan->get_fetch_cur_time() && !pctx->has_cur_time()) {
        pctx->set_cur_time(ObTimeUtility::current_time(), *(sql_ctx.session_info_));
      }
      pctx->set_last_trace_id(sql_ctx.session_info_->get_last_trace_id());
    }

    ObSQLSessionInfo *session_info = result.get_exec_context().get_my_session();
    BEGIN_OPT_TRACE(session_info, ObString(strlen(sql_ctx.sql_id_), sql_ctx.sql_id_));
    OPT_TRACE_ENV;
    OPT_TRACE_SESSION_INFO;
    OPT_TRACE_PARAMETERS;
    OPT_TRACE_TITLE("CURRENT SQL TEXT");
    OPT_TRACE("sql_id =", ObString(strlen(sql_ctx.sql_id_), sql_ctx.sql_id_));
    OPT_TRACE(ObString(parse_result.input_sql_len_, parse_result.input_sql_));

    // hint seed injected random status rand
    const ObQueryHint &query_hint = stmt->get_query_ctx()->get_query_hint();
    if (query_hint.has_outline_data() || session_info->is_inner()){
      // if there is outline data no error inject
    } else {
      int64_t tmp_ret = (OB_E(EventTable::EN_GENERATE_RANDOM_PLAN) OB_SUCCESS);
      if (OB_SUCCESS == tmp_ret) {
        // do nothing
      } else {
        time_t seed = OB_ERROR == tmp_ret ? time(NULL) : std::abs(tmp_ret);
        stmt->get_query_ctx()->set_injected_random_status(true);
        stmt->get_query_ctx()->set_random_plan_seed(seed);
        LOG_INFO("The random seed for plan gen is ", K(seed));
        OPT_TRACE("The random seed for plan gen is ", seed);
      }
    }

    if (OB_FAIL(ret)) {
    } else if (OB_FAIL(transform_stmt(&stmt->get_query_ctx()->sql_schema_guard_,
                                      opt_stat_mgr_,
                                      &self_addr_,
                                      phy_plan,
                                      result.get_exec_context(),
                                      stmt))) {
    } else if (OB_FAIL(generate_stmt_with_reconstruct_sql(stmt,
                                                          pc_ctx,
                                                          sql_ctx,
                                                          result,
                                                          phy_plan))) {
    } else if (OB_FALSE_IT(optctx.set_root_stmt(stmt))) {
    } else if (OB_FAIL(optimize_stmt(optimizer, *(sql_ctx.session_info_),
                                      *stmt, logical_plan))) {
    } else if (OB_FAIL(create_expr_constraints(*stmt->get_query_ctx(),
                                                result.get_exec_context()))){
    } else if (OB_ISNULL(logical_plan)) {
      ret = OB_INVALID_ARGUMENT;
    } else if (OB_FAIL(code_generate(sql_ctx, result, stmt,
                                     stmt_need_privs,
                                     logical_plan, phy_plan))) {
    } else if (OB_FAIL(prepare_outline_for_phy_plan(logical_plan,
                                                    phy_plan))) {
    } else if (logical_plan->get_optimizer_context().is_online_ddl()) {
      int tmp_ret = OB_SUCCESS;
      ObExplainDisplayOpt option;
      option.with_tree_line_ = false;
      ObSqlPlan sql_plan(logical_plan->get_allocator());
      ObSEArray<common::ObString, 32> plan_strs;
      if (OB_TMP_FAIL(sql_plan.print_sql_plan(logical_plan->get_plan_root(),
                                          EXPLAIN_EXTENDED,
                                          option,
                                          plan_strs))) {
      } else {
        _OB_LOG(INFO, "ddl sql:%.*s", parse_result.input_sql_len_, parse_result.input_sql_);
        LOG_INFO("ddl plan");
        for (int64_t i = 0; OB_SUCCESS == tmp_ret && i < plan_strs.count(); i++) {
          _OB_LOG(INFO, "%.*s", plan_strs.at(i).length(), plan_strs.at(i).ptr());
        }
        LOG_INFO("ddl stmt:", KPC(stmt));
      }
    }
    END_OPT_TRACE(session_info);
    if (OB_SUCC(ret)) {
      ObSqlPlan sql_plan(result.get_mem_pool());
      if (stmt->is_explain_stmt()) {
        // do nothing
      } else if (OB_FAIL(sql_plan.store_sql_plan(logical_plan, phy_plan))) {
      } else {
        phy_plan->set_record_plan_info(true);
      }
    }

    // memory debug for sql work arena
    if (OB_UNLIKELY(result.get_mem_pool().total() > SQL_MEM_SIZE_LIMIT)) {
      int64_t total_mem_used = result.get_mem_pool().total();
      LOG_INFO("[SQL MEM USAGE] use too much memory", K(total_mem_used),
                K(ObString(parse_result.input_sql_len_, parse_result.input_sql_)));
    }

    if (OB_FAIL(ret)) {
    } else if (OB_ISNULL(stmt->get_query_ctx())) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      result.get_exec_context().reference_my_plan(phy_plan);
    }
    } // end HEAP_VAR optctx
  }
  return ret;
}

int ObSql::generate_stmt_with_reconstruct_sql(ObDMLStmt* &stmt,
                                              ObPlanCacheCtx *pc_ctx,
                                              ObSqlCtx &sql_ctx,
                                              ObResultSet &result,
                                              ObPhysicalPlan *phy_plan)
{
  int ret = OB_SUCCESS;
  ObString sql;
  ObObjPrintParams print_param;
  ObSQLSessionInfo *session = sql_ctx.session_info_;
  ObPhysicalPlanCtx *phy_plan_ctx = NULL;
  if (OB_ISNULL(session) || OB_ISNULL(pc_ctx) ||
      OB_ISNULL(stmt) || (OB_ISNULL(stmt->get_query_ctx())) ||
      OB_ISNULL(phy_plan_ctx=pc_ctx->exec_ctx_.get_physical_plan_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if ((OB_E(EventTable::EN_GENERATE_PLAN_WITH_RECONSTRUCT_SQL) OB_SUCCESS) == OB_SUCCESS) {
    //do nothing
  } else if (!session->is_user_session()) {
    //do nothing
  } else if (OB_FAIL(ObSQLUtils::reconstruct_sql(pc_ctx->allocator_,
                                                stmt,
                                                sql,
                                                sql_ctx.schema_guard_,
                                                print_param,
                                                &phy_plan_ctx->get_param_store(),
                                                session))) {
  } else {
    ObStmt *basic_stmt = NULL;
    ParseResult parse_result;
    ObParser parser(pc_ctx->allocator_,
                    session->get_sql_mode(),
                    session->get_charsets4parser());
    stmt->get_query_ctx()->global_dependency_tables_.reuse();
    if (OB_FAIL(parser.parse(sql, parse_result))) {
    } else if (OB_FAIL(generate_stmt(parse_result,
                                    pc_ctx,
                                    sql_ctx,
                                    result.get_mem_pool(),
                                    result,
                                    basic_stmt,
                                    NULL))) {
    } else if (OB_ISNULL(basic_stmt) || !basic_stmt->is_dml_stmt()) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FALSE_IT(stmt = static_cast<ObDMLStmt*>(basic_stmt))) {
    } else if (OB_FAIL(transform_stmt(&stmt->get_query_ctx()->sql_schema_guard_,
                                      opt_stat_mgr_,
                                      &self_addr_,
                                      phy_plan,
                                      result.get_exec_context(),
                                      stmt,
                                      true))) {
    }
  }
  return ret;
}

int ObSql::prepare_outline_for_phy_plan(ObLogPlan *logical_plan,
                                        ObPhysicalPlan *phy_plan)
{
  int ret = OB_SUCCESS;
  void *tmp_ptr = NULL;
  char *buf = NULL;
  PlanText plan_text;

  if (OB_ISNULL(logical_plan) || OB_ISNULL(phy_plan)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_UNLIKELY(NULL == (tmp_ptr = logical_plan->get_allocator().alloc(OB_MAX_SQL_LENGTH)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
    LOG_ERROR("fail to alloc memory", K(ret));
  } else if (FALSE_IT(buf = static_cast<char *>(tmp_ptr))) {
  } else {
    //print outline data
    plan_text.buf_ = buf;
    plan_text.buf_len_ = OB_MAX_SQL_LENGTH;
    if (OB_FAIL(ObSqlPlan::get_plan_outline_info_one_line(plan_text, logical_plan))) {
    } else if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                       ObString(plan_text.pos_, plan_text.buf_),
                                       phy_plan->stat_.outline_data_))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else {
    //print used hint
    plan_text.pos_ = 0;
    if (OB_FAIL(ObSqlPlan::get_plan_used_hint_info_one_line(plan_text, logical_plan))) {
    } else if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                       ObString(plan_text.pos_, plan_text.buf_),
                                       phy_plan->stat_.hints_info_))) {
    }
  }
  //free temp buffer
  if (NULL != tmp_ptr && NULL != logical_plan) {
    logical_plan->get_allocator().free(tmp_ptr);
  }
  return ret;
}
// stmt full const folding.
// To rewrite layer const folding extracted function
int ObSql::calc_pre_calculable_exprs(
    ObIArray<ObHiddenColumnItem> &calculable_exprs,
    ObExecContext &exec_ctx,
    ObDMLStmt &stmt,
    ObPhysicalPlan &phy_plan)
{
  int ret = OB_SUCCESS;
  stmt::StmtType stmt_type = stmt::T_NONE;
  ObDList <ObSqlExpression> pre_calc_exprs;
  ObPhysicalPlanCtx *plan_ctx = exec_ctx.get_physical_plan_ctx();
  ObSQLSessionInfo *session_info = exec_ctx.get_my_session();
  ObRawExprFactory *expr_factory = exec_ctx.get_expr_factory();
  if (OB_ISNULL(plan_ctx) || OB_ISNULL(session_info) || OB_ISNULL(expr_factory)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (stmt.is_explain_stmt()) {
    ObDMLStmt *real_stmt = static_cast<ObExplainStmt&>(stmt).get_explain_query_stmt();
    if (OB_ISNULL(real_stmt)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      stmt_type = real_stmt->get_stmt_type();
    }
  } else {
    stmt_type = stmt.get_stmt_type();
  }
  for (int64_t i = 0; OB_SUCC(ret) && i < calculable_exprs.count(); i++) {
    bool transformed = false;
    if (OB_FAIL(ObTransformPreProcess::transform_expr(*expr_factory,
                                                      *session_info,
                                                      calculable_exprs.at(i).expr_,
                                                      transformed))) {
    }
  }
  if (OB_SUCC(ret)) {
    // is_ignore should be set before pre_calculate, which is used in column_conv
    bool is_ignore_stmt = false;
    bool need_fetch_cur_time = false;
    ObDelUpdStmt *modify_stmt = dynamic_cast<ObDelUpdStmt*>(&stmt);
    if (NULL != modify_stmt) {
      is_ignore_stmt = modify_stmt->is_ignore();
    }

    if (OB_FAIL(calc_pre_calculable_exprs(stmt, calculable_exprs,
                                          is_ignore_stmt, exec_ctx, phy_plan))) {
    }
  }
  return ret;
}

int ObSql::transform_stmt(ObSqlSchemaGuard *sql_schema_guard,
                          common::ObOptStatManager *opt_stat_mgr,
                          common::ObAddr *self_addr,
                          ObPhysicalPlan *phy_plan,
                          ObExecContext &exec_ctx,
                          ObDMLStmt *&stmt,
                          bool ignore_trace_event)
{
  int ret = OB_SUCCESS;
  ACTIVE_SESSION_FLAG_SETTER_GUARD(in_rewrite);
  ObDMLStmt *transform_stmt = stmt;
  int64_t last_mem_usage = exec_ctx.get_allocator().total();
  int64_t transformer_mem_usage = 0;
  //get transform stmt
  if (OB_ISNULL(stmt) || OB_ISNULL(stmt->get_query_ctx()) || OB_ISNULL(sql_schema_guard) ||
      OB_ISNULL(sql_schema_guard->get_schema_guard()) ||
      OB_ISNULL(opt_stat_mgr) || OB_ISNULL(self_addr)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_ISNULL(exec_ctx.get_my_session())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (stmt->is_explain_stmt()) {
    if (OB_ISNULL(transform_stmt = static_cast<ObExplainStmt*>(stmt)->get_explain_query_stmt())) {
      ret = OB_ERR_UNEXPECTED;
      LOG_ERROR("Explain query stmt is NULL", K(ret));
    }
  } else { }//do nothing

  ObTransformerCtx trans_ctx;
  ObSchemaChecker schema_checker;

  if (OB_FAIL(ret)) {
  } else if (OB_FAIL(schema_checker.init(*sql_schema_guard))) {
  } else {
    trans_ctx.allocator_ = &exec_ctx.get_allocator();
    trans_ctx.schema_checker_ = &schema_checker;
    trans_ctx.session_info_ = exec_ctx.get_my_session();
    trans_ctx.exec_ctx_ = &exec_ctx;
    trans_ctx.expr_factory_ = exec_ctx.get_expr_factory();
    trans_ctx.stmt_factory_ = exec_ctx.get_stmt_factory();
    trans_ctx.opt_stat_mgr_ = opt_stat_mgr;
    trans_ctx.sql_schema_guard_ = sql_schema_guard;
    trans_ctx.self_addr_ = self_addr;
    // trans_ctx.merged_version_ = merged_version;

    trans_ctx.phy_plan_ = phy_plan;
  }

  NG_TRACE(transform_begin);
  OPT_TRACE_TITLE("START TRANSFORM");
  bool need_transform = transform_stmt->is_valid_transform_stmt();
  if (ignore_trace_event) {
    //do nothing
  } else if ((OB_E(EventTable::EN_GENERATE_PLAN_WITH_RECONSTRUCT_SQL)OB_SUCCESS) != OB_SUCCESS) {
    if (transform_stmt->is_dml_write_stmt()) {
      need_transform = false;
    }
  }
  if (OB_SUCC(ret) && need_transform) {
    ObTransformerImpl transformer(&trans_ctx);
    if (OB_FAIL(transformer.transform(transform_stmt))) {
    } else if (stmt->is_explain_stmt()) {
      static_cast<ObExplainStmt*>(stmt)->set_explain_query_stmt(transform_stmt);
    } else {
      bool or_expansion_happened = false;
      if (OB_FAIL(transformer.get_cost_based_trans_happened(OR_EXPANSION,
                                                            or_expansion_happened))) {
      } else if (or_expansion_happened) {
        if (OB_ISNULL(exec_ctx.get_physical_plan_ctx())) {
          ret = OB_INVALID_ARGUMENT;
        } else {
          exec_ctx.get_physical_plan_ctx()->set_or_expand_transformed(true);
        }
      }
      if (OB_SUCC(ret)) {
        stmt = transform_stmt;
      }
    }
  }

  if (OB_SUCC(ret)) {
    OPT_TRACE("after transform, sql is\n", transform_stmt);
    OPT_TRACE_TIME_USED;
    OPT_TRACE_MEM_USED;
  }
  transformer_mem_usage = exec_ctx.get_allocator().total() - last_mem_usage;
  return ret;
}

int ObSql::optimize_stmt(
    ObOptimizer &optimizer,
    const ObSQLSessionInfo &session_info,
    ObDMLStmt &stmt,
    ObLogPlan *&logical_plan)
{
  int ret = OB_SUCCESS;
  logical_plan = NULL;
  OPT_TRACE_TITLE("START GENERATE PLAN");
  if (OB_FAIL(optimizer.optimize(stmt, logical_plan))) {
  } else if (OB_FAIL(optimizer.update_column_usage_infos())) {
  } else if (OB_ISNULL(logical_plan)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    OPT_TRACE_TIME_USED;
    OPT_TRACE_MEM_USED;
    OPT_TRACE_TITLE("SYSTEM STATS:");
    OPT_TRACE(logical_plan->get_optimizer_context().get_system_stat());
    OPT_TRACE(logical_plan);
  }

  return ret;
}

int ObSql::code_generate(
    ObSqlCtx &sql_ctx,
    ObResultSet &result,
    ObDMLStmt *stmt,
    ObStmtNeedPrivs &stmt_need_privs,
    ObLogPlan *logical_plan,
    ObPhysicalPlan *&phy_plan)
{
  int ret = OB_SUCCESS;
  int64_t last_mem_usage = 0;
  int64_t codegen_mem_usage = 0;
  ObPhysicalPlanCtx *pctx = result.get_exec_context().get_physical_plan_ctx();
  if (OB_ISNULL(stmt) || OB_ISNULL(logical_plan) || OB_ISNULL(stmt->get_query_ctx())
        || OB_ISNULL(phy_plan) || OB_ISNULL(sql_ctx.session_info_)
        || OB_ISNULL(pctx)) {
      ret = OB_INVALID_ARGUMENT;
  } else {
    ObCodeGenerator code_generator(&(pctx->get_datum_param_store()));
    phy_plan->set_is_packed(logical_plan->get_optimizer_context().is_packed());
    if (OB_FAIL(code_generator.generate(*logical_plan, *phy_plan))) {
    } else {
      // session's ignore_stmt status for CG use, needs to be cleared after CG ends
      sql_ctx.session_info_->set_ignore_stmt(false);
      last_mem_usage = phy_plan->get_mem_size();
    }
  }
  //add local_session_var array to phy_plan_ctx
  if (OB_SUCC(ret)) {
    if (OB_FAIL(phy_plan->set_all_local_session_vars(sql_ctx.all_local_session_vars_))) {
    }
  }
  NG_TRACE(cg_end);

  // set phy table location in task_exec_ctx, query_timeout in exec_context
  if (OB_SUCC(ret)) {
    const ObPlanCachePolicy plan_cache_policy = phy_plan->get_phy_plan_hint().plan_cache_policy_;
    ObPhyPlanHint phy_hint(logical_plan->get_optimizer_context().get_global_hint());
    if (OB_USE_PLAN_CACHE_NONE == plan_cache_policy) {
      phy_hint.plan_cache_policy_ = OB_USE_PLAN_CACHE_NONE;
    }
    // set larger query_time for IS
    if (stmt->get_query_ctx()->has_is_table_) {
      int tmp_ret = OB_SUCCESS;
      if (phy_hint.query_timeout_ <= 0) {
        if (OB_SUCCESS != (tmp_ret = sql_ctx.session_info_->get_query_timeout(
                    phy_hint.query_timeout_))) {
        }
      }
      phy_hint.query_timeout_ *= 10;
    } else {}//do nothing
    ObSEArray<ObTablePartitionInfo *, 3> tbl_part_infos;
    phy_plan->set_phy_plan_hint(phy_hint);//remember in phy_plan
    if (OB_FAIL(ret)) {
      // do nothing
    } else {
      if (OB_FAIL(logical_plan->get_global_table_partition_info(tbl_part_infos))) {
      } else if (OB_FAIL(sql_ctx.set_partition_infos(
              tbl_part_infos,
              result.get_exec_context().get_allocator()))) {
      } else {
        ObDASCtx &das_ctx = DAS_CTX(result.get_exec_context());
        for (int64_t i = 0; OB_SUCC(ret) && i < tbl_part_infos.count(); i++) {
          ObTableLocation &tl = tbl_part_infos.at(i)->get_table_location();
          if (!tl.use_das()) {
            const ObCandiTableLoc &candi_table_loc = tbl_part_infos.at(i)->get_phy_tbl_location_info();
            if (OB_FAIL(das_ctx.add_candi_table_loc(tl.get_loc_meta(), candi_table_loc))) {
            } else if (OB_UNLIKELY(das_ctx.get_table_loc_list().empty())) {
              ret = OB_ERR_UNEXPECTED;
            }
          }
        } // for end
      }
    }
    // get tablet id and partition id for non partition tables
    if (OB_SUCC(ret)) {
      bool skip_non_partition_optimized = false;
      for (int64_t i = 0; OB_SUCC(ret) && i < tbl_part_infos.count(); i++) {
        ObTableLocation &tl = tbl_part_infos.at(i)->get_table_location();
        if (tl.is_partitioned() || is_virtual_table(tl.get_loc_meta().ref_table_id_)) {
          skip_non_partition_optimized = true;
          break;
        }
      }
      if (OB_SUCC(ret) && !skip_non_partition_optimized) {
        for (int64_t i = 0; OB_SUCC(ret) && i < tbl_part_infos.count(); i++) {
          ObTableLocation &tl = tbl_part_infos.at(i)->get_table_location();
          if (OB_FAIL(tl.calc_not_partitioned_table_ids(result.get_exec_context()))) {
          } else {
            tl.set_is_non_partition_optimized(true);
          }
        }
      }
    }
    // set table location for phy_plan
    if (OB_SUCC(ret)) {
      if (OB_FAIL(phy_plan->set_table_locations(tbl_part_infos, *sql_ctx.schema_guard_))) {
      }
    }
  }

  // set multi stmt info
  if (OB_SUCC(ret)) {
    if (OB_FAIL(sql_ctx.set_multi_stmt_rowkey_pos(logical_plan->get_multi_stmt_rowkey_pos(),
                                                  result.get_exec_context().get_allocator()))) {
    } else {
    }
  }

  if (OB_SUCC(ret)) {
    bool use_plan_cache = sql_ctx.session_info_->get_local_ob_enable_plan_cache();
    ObPlanCache *plan_cache = plan_cache_;
    if (OB_UNLIKELY(NULL == plan_cache)) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      if (use_plan_cache) {
        if (OB_FAIL(phy_plan->set_stmt_need_privs(stmt_need_privs))) {
        }
      }
      if (OB_SUCC(ret)) {
        if (OB_FAIL(phy_plan->init_operator_stats())) {
        } else {
          codegen_mem_usage = phy_plan->get_mem_size() - last_mem_usage;
        }
      }
    }
  }

  return ret;
}

inline int ObSql::sanity_check(ObSqlCtx &context)
{
  int ret = OB_SUCCESS;
  if (!inited_) {
  } else if (OB_UNLIKELY(NULL == context.session_info_)
             || OB_UNLIKELY(NULL == context.schema_guard_)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    // do nothing
  }
  return ret;
}

int ObSql::init_result_set(ObSqlCtx &context, ObResultSet &result_set)
{
  return init_exec_context(context, result_set.get_exec_context());
}

OB_INLINE int ObSql::init_exec_context(const ObSqlCtx &context, ObExecContext &exec_ctx)
{
  int ret = OB_SUCCESS;
  ObSqlExecutorCtx &task_exec_ctx = exec_ctx.get_sql_exec_ctx();
  task_exec_ctx.set_retry_times(context.retry_times_);
  if (OB_FAIL(exec_ctx.create_physical_plan_ctx())) {
  } else {
    exec_ctx.set_my_session(context.session_info_);
    bind_exec_context_runtime_services(exec_ctx);
    exec_ctx.set_sql_ctx(const_cast<ObSqlCtx*>(&context));
    if (OB_NOT_NULL(exec_ctx.get_physical_plan_ctx()) && OB_NOT_NULL(context.session_info_)) {
      int64_t query_timeout = 0;
      context.session_info_->get_query_timeout(query_timeout);
      exec_ctx.get_physical_plan_ctx()->set_timeout_timestamp(
        context.session_info_->get_query_start_time() + query_timeout);
      exec_ctx.set_retry_info(&context.session_info_->get_retry_info());
    }
  }
  return ret;
}

int ObSql::execute_get_plan(ObPlanCache &plan_cache,
                            ObPlanCacheCtx &pc_ctx,
                            ObCacheObjGuard& guard)
{
  int ret = OB_SUCCESS;
  ObIAllocator &allocator = pc_ctx.allocator_;
  ObSQLSessionInfo *session = pc_ctx.sql_ctx_.session_info_;
  ObPhysicalPlanCtx *pctx = pc_ctx.exec_ctx_.get_physical_plan_ctx();
  if (OB_ISNULL(session) || OB_ISNULL(pctx)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    if (PC_PS_MODE == pc_ctx.mode_ || PC_PL_MODE == pc_ctx.mode_) {
      // TODO change pl mode hit cache as text mode.
      ObPsStmtId stmt_id = pc_ctx.fp_result_.pc_key_.key_id_;
      if (OB_FAIL(plan_cache.get_ps_plan(guard, stmt_id, pc_ctx))) {
        if (OB_SQL_PC_NOT_EXIST == ret || OB_PC_LOCK_CONFLICT == ret) {
          // do nothing
        } else {
        }
      }
    } else {
      if (OB_FAIL(plan_cache.get_plan(allocator, pc_ctx, guard))) {
        if (OB_SQL_PC_NOT_EXIST == ret || OB_PC_LOCK_CONFLICT == ret) {
          // do nothing
        } else {
        }
      }
    }
  }
  return ret;
}

int ObSql::pc_get_plan_and_fill_result(ObPlanCacheCtx &pc_ctx,
                                       ObResultSet &result,
                                       int &get_plan_err,
                                       bool &need_disconnect)
{
  UNUSED(need_disconnect);
  int ret = OB_SUCCESS;
  ObPhysicalPlan *plan = NULL;
  ObExecContext &exec_ctx = result.get_exec_context();
  ObCacheObjGuard& guard = result.get_cache_obj_guard();

  if (OB_FAIL(pc_get_plan(pc_ctx, guard, get_plan_err,
                          exec_ctx.get_need_disconnect_for_update()))) {
  } else if (OB_SUCCESS != get_plan_err) {
    //get plan from plan cache failed
  } else if ( FALSE_IT(plan = static_cast<ObPhysicalPlan*>(guard.get_cache_obj()))) {
    // do nothing
  } else if (OB_ISNULL(plan)) {
    ret = OB_ERR_UNEXPECTED;
  } else {
    result.set_is_from_plan_cache(true);
    if (OB_FAIL(result.from_plan(*plan, pc_ctx.fp_result_.raw_params_))) {
    }
  }
  return ret;
}

int ObSql::pc_get_plan(ObPlanCacheCtx &pc_ctx,
                       ObCacheObjGuard& guard,
                       int &get_plan_err,
                       bool &need_disconnect)
{
  ACTIVE_SESSION_FLAG_SETTER_GUARD(in_get_plan_cache);
  int ret = OB_SUCCESS;
  //NG_TRACE(cache_get_plan_begin);
  ObPlanCache *plan_cache = NULL;
  if (OB_ISNULL(plan_cache = plan_cache_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(execute_get_plan(*plan_cache, pc_ctx, guard))) {
    if (OB_EAGAIN == ret
        || OB_ARRAY_BINDING_ROLLBACK == ret
        || OB_BATCHED_MULTI_STMT_ROLLBACK == ret) {
      /*do nothing*/
    } else if (OB_PC_LOCK_CONFLICT == ret) {
      // Lock contention is a plan-cache miss. Continue on the long path instead
      // of exposing an internal cache error to the client.
      get_plan_err = ret;
      ret = OB_SUCCESS;
    } else {
      get_plan_err = ret;
      int tmp_ret = OB_SUCCESS;
      tmp_ret = OB_E(EventTable::EN_PC_NOT_SWALLOW_ERROR) OB_SUCCESS;
      if (OB_SUCCESS != tmp_ret) {
         // do nothing
        if (OB_SQL_PC_NOT_EXIST == ret || OB_REACH_MEMORY_LIMIT == ret) {
          ret = OB_SUCCESS;
        }
      } else {
        ret = OB_SUCCESS; // get plan error, cover error code, ensure that errors from plan cache do not affect the normal execution path
      }
    }
  } else { // get plan successfully
    plan_cache->inc_hit_and_access_cnt();
    ObPhysicalPlan* plan = static_cast<ObPhysicalPlan*>(guard.get_cache_obj());
    if (OB_ISNULL(plan)) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("fail to get plan cache");
    } else {
      // Hit the plan cache, then it cannot be a commit or rollback statement, default to keep connecting
      need_disconnect = false;
      //FIXME qianfu NG_TRACE_EXT(set_need_disconnect, OB_ID(need_disconnect), false);
      pc_ctx.sql_ctx_.plan_cache_hit_ = true;
      // In extreme performance scenarios (perf_event=true), permission information is no longer validated
      if (OB_SUCC(ret) && config::enable_perf_event()) {
        if (OB_FAIL(ObPrivilegeCheck::check_read_only(pc_ctx.sql_ctx_, plan->get_stmt_type(), false,
                                                      plan->get_stmt_need_privs()))) {
        } else if (OB_FAIL(ObPrivilegeCheck::check_privilege(
                                          pc_ctx.sql_ctx_,
                                          plan->get_stmt_need_privs()))) {
        } else {
          NG_TRACE(check_priv);
        }
        if (OB_SUCC(ret) && OB_FAIL(ObPrivilegeCheck::check_password_expired(pc_ctx.sql_ctx_,
                                                                            stmt::T_NONE))) {
          LOG_WARN("Falied to check password expired", K(ret));
        }
      }

    }
  }
  return ret;
}

int ObSql::get_outline_data(ObSqlCtx &context,
                            ObPlanCacheCtx &pc_ctx,
                            const ObString &signature_sql,
                            const ObString &signature_format_sql,
                            ObOutlineState &outline_state,
                            ParseResult &outline_parse_result)
{
  NG_TRACE(transform_with_outline_begin);
  int ret = OB_SUCCESS;
  memset(&outline_parse_result, 0, sizeof(ParseResult));
  ObString outline_content;
  ObSQLSessionInfo *session = pc_ctx.sql_ctx_.session_info_;
  if (OB_ISNULL(session)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (0 != context.first_plan_hash_) {
    outline_content = context.first_outline_data_;
  } else if (OB_INVALID_ID == context.plan_key_.db_id_
             || context.is_batch_params_execute()) {
    //no outline is available when database name of session is not specified, just keep the stmt
  } else if (pc_ctx.is_begin_commit_stmt()) {
    /* do nothing */
  } else if (OB_FAIL(get_outline_data(pc_ctx, signature_sql, signature_format_sql, outline_state, outline_content))) {
  }

  if (OB_SUCC(ret) && !outline_content.empty()) {
    ObParser parser(pc_ctx.allocator_, session->get_sql_mode(), session->get_charsets4parser());
    ObSqlString sql_helper;
    ObString temp_outline_sql;
    if (OB_FAIL(sql_helper.assign_fmt("select %.*s 1 from dual", outline_content.length(),
                                                                 outline_content.ptr()))) {
    } else if (OB_FAIL(ob_write_string(pc_ctx.allocator_, sql_helper.string(), temp_outline_sql))) {
    } else if (OB_FAIL(parser.parse(temp_outline_sql, outline_parse_result))) {
      outline_state.reset();
      ret = OB_SUCCESS;
    }
  }
  NG_TRACE(transform_with_outline_end);
  return ret;
}

int ObSql::get_outline_data(ObPlanCacheCtx &pc_ctx,
                            const ObString &signature_sql,
                            const ObString &signature_format_sql,
                            ObOutlineState &outline_state,
                            ObString &outline_content)
{
  int ret = OB_SUCCESS;
  const ObOutlineInfo *outline_info = NULL;
  ObSchemaGetterGuard *schema_guard = pc_ctx.sql_ctx_.schema_guard_;
  const uint64_t database_id = pc_ctx.sql_ctx_.plan_key_.db_id_;
  const ObString sql_id = pc_ctx.sql_ctx_.plan_key_.sql_id_;
  const ObString format_sql_id = pc_ctx.sql_ctx_.plan_key_.format_sql_id_;
  ObSQLSessionInfo *session = pc_ctx.sql_ctx_.session_info_;
  outline_state.reset();
  int64_t schema_version = OB_INVALID_VERSION;
  if (OB_ISNULL(session) || OB_ISNULL(schema_guard)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(schema_guard->get_schema_version(schema_version))) {
  } else if (OB_CORE_SCHEMA_VERSION >= schema_version) {
    // local schema is fall behind, do not use outline
  } else {
    char *buf = NULL;
    int64_t pos = 0;
    int64_t format_pos = 0;
    int64_t size = signature_sql.get_serialize_size() + signature_format_sql.get_serialize_size();
    ObString outline_key;
    ObString format_outline_key;
    ObIAllocator &allocator = CURRENT_CONTEXT->get_arena_allocator();
    if (0 == size) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_ISNULL(buf = (char *)allocator.alloc(size))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      LOG_ERROR("fail to alloc mem", K(ret));
    } else if (OB_FAIL(signature_sql.serialize(buf, size, pos))) {
    } else if (OB_FALSE_IT(outline_key.assign_ptr(buf, static_cast<ObString::obstr_size_t>(pos)))) {
    } else if (!signature_format_sql.empty()
        && OB_FAIL(signature_format_sql.serialize(buf + pos, size, format_pos))) {
    } else if (!signature_format_sql.empty()
        && FALSE_IT(format_outline_key.assign_ptr(buf + pos,
            static_cast<ObString::obstr_size_t>(format_pos)))) {
      // try normal outline
    } else if (OB_FAIL(schema_guard->get_outline_info_with_signature(database_id,
                                                                     outline_key,
                                                                     false, /*normal outline*/
                                                                     outline_info))) {
      ret = OB_SUCCESS;
    // try normal outline
    } else if (NULL == outline_info &&
               OB_FAIL(schema_guard->get_outline_info_with_sql_id(database_id,
                                                                sql_id,
                                                                false, /*normal outline*/
                                                                outline_info))) {
      ret = OB_SUCCESS;
    // try format outline
    } else if (NULL == outline_info && !format_outline_key.empty() &&
              OB_FAIL(schema_guard->get_outline_info_with_signature(database_id,
                                                                     format_outline_key,
                                                                     true, /*format outline*/
                                                                     outline_info))) {
      ret = OB_SUCCESS;

    // try format outline
    } else if (NULL == outline_info && !format_sql_id.empty() &&
               OB_FAIL(schema_guard->get_outline_info_with_sql_id(database_id,
                                                                format_sql_id,
                                                                true, /*format outline*/
                                                                outline_info))) {
      ret = OB_SUCCESS;
    }
  }

  if (OB_SUCC(ret) && NULL != outline_info) {
    ObString outline_content_copy = outline_info->get_outline_content_str();
    if (OB_FAIL(ObSQLUtils::convert_sql_text_from_schema_for_resolve(pc_ctx.allocator_,
                                                                     session->get_dtc_params(),
                                                                     outline_content_copy))) {
      //outline_content is stored using UTF8, if client sql is GBK, convert may failed
      outline_content = ObString::make_empty_string();
      ret = OB_SUCCESS;
    } else {
      outline_content = outline_content_copy;
      outline_state.is_plan_fixed_ = !outline_info->get_outline_content_str().empty();
      outline_state.outline_version_.object_id_ = outline_info->get_outline_id();
      outline_state.outline_version_.version_ = outline_info->get_schema_version();
      outline_state.outline_version_.object_type_ = DEPENDENCY_OUTLINE;
    }
  }
  return ret;
}

int ObSql::parser_and_check(const ObString &outlined_stmt,
                            ObExecContext &exec_ctx,
                            ObPlanCacheCtx &pc_ctx,
                            ParseResult &parse_result,
                            int get_plan_err,
                            bool &add_plan_to_pc,
                            bool &is_enable_transform_tree)
{
  int ret = OB_SUCCESS;
  ObIAllocator &allocator = pc_ctx.allocator_;
  ObSQLSessionInfo *session = exec_ctx.get_my_session();
  ObPhysicalPlanCtx *pctx = exec_ctx.get_physical_plan_ctx();
  bool is_stack_overflow = false;
  bool is_show_variables = false;
  bool is_explain_parameterize = false;
  stmt::StmtType stmt_type = stmt::T_NONE;
  if (OB_ISNULL(session)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(check_stack_overflow(is_stack_overflow))) {
  } else if (is_stack_overflow) {
    ret = OB_SIZE_OVERFLOW;
  } else if (OB_ISNULL(pctx)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    pctx->reset_datum_param_store();
    pctx->get_param_store_for_update().reuse();
    ObParser parser(allocator, session->get_sql_mode(), session->get_charsets4parser());
    if (OB_FAIL(parser.parse(outlined_stmt, parse_result,
                             STD_MODE,
                             pc_ctx.sql_ctx_.handle_batched_multi_stmt(),
                             false, NULL != session->get_pl_context()))) {
    } else if ((PC_PS_MODE == pc_ctx.mode_ || PC_PL_MODE == pc_ctx.mode_)
      && OB_FAIL(construct_param_store_from_parameterized_params(
                    pc_ctx, pctx->get_param_store_for_update()))) {
    }
    pc_ctx.sql_ctx_.is_sensitive_ |= parse_result.contain_sensitive_data_;
    if (OB_SUCC(ret)) {
      // parser returns success
      if (OB_ISNULL(parse_result.result_tree_)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_ERROR("parse result tree is NULL", K(ret));
      } else if (OB_UNLIKELY(parse_result.result_tree_->num_child_ < 1)) {
        ret = OB_ERR_UNEXPECTED;
        LOG_ERROR("child count of parse result tree is less than 1",
                  K(ret), K(parse_result.result_tree_->num_child_));
      } else if (OB_ISNULL(parse_result.result_tree_->children_[0])) {
        ret = OB_ERR_UNEXPECTED;
        LOG_ERROR("children_[0] ofparse result tree is NULL",
                  K(ret), K(parse_result.result_tree_->num_child_));
      } else {
        ObItemType parse_stmt_type = parse_result.result_tree_->children_[0]->type_;
        if (T_COMMIT == parse_stmt_type || T_ROLLBACK == parse_stmt_type) {
          // Is commit or rollback statement, default disconnect
        } else {
          // Not a commit or rollback statement, default to keep connecting
          exec_ctx.set_need_disconnect(false);
          //FIXME qianfu NG_TRACE_EXT(set_need_disconnect, OB_ID(need_disconnect), false);
        }
      }
    } else if (!ObSQLUtils::check_need_disconnect_parser_err(ret)) {
      exec_ctx.set_need_disconnect(false);
    } else {
      // parser returns an unknown error code, need to disconnect from the client
    }
    if (OB_SUCC(ret)) {
      stmt::StmtType stmt_type = stmt::T_NONE;
      if (OB_ISNULL(parse_result.result_tree_)) {
        ret = OB_INVALID_ARGUMENT;
      } else if (OB_ISNULL(parse_result.result_tree_->children_)
                 || OB_UNLIKELY(parse_result.result_tree_->num_child_ < 1)) {
        ret = OB_INVALID_ARGUMENT;
      } else {
        ParseNode *children_node = parse_result.result_tree_->children_[0];
        if (OB_ISNULL(children_node)) {
          ret = OB_INVALID_ARGUMENT;
        // In addition to ordinary dml stmt, ? in explain stmt also needs to be judged here
        } else if (!(PC_PS_MODE == pc_ctx.mode_ || PC_PL_MODE == pc_ctx.mode_)
                   && (children_node->type_ == T_EXPLAIN || IS_DML_STMT(children_node->type_))
                   && (children_node->value_ > 0)) {
          ret = OB_ERR_PARSE_SQL;//children_node->value_ > 0, indicates that it has a question_mark
          const char *err_msg = "?";
          int32_t str_len = static_cast<int32_t>(strlen(err_msg));
          int32_t line_no = 1;
          LOG_USER_ERROR(OB_ERR_PARSE_SQL, ob_errpkt_strerror(OB_ERR_PARSER_SYNTAX), str_len, err_msg, line_no);
        } else if (OB_FAIL(ObResolverUtils::resolve_stmt_type(parse_result, stmt_type))) {
        } else {
          ObItemType type = children_node->type_;
          // If it is not a DML statement, then do not enter plan cache
          ObPlanCache *plan_cache = NULL;
          if (T_SHOW_VARIABLES == type) {
            is_show_variables = true;
          }
          if (ObStmt::is_ddl_stmt(stmt_type, true)) {
            THIS_WORKER.set_timeout_ts(session->get_query_start_time() + config::_ob_ddl_timeout());
          }
          if (IS_DML_STMT(type) || is_show_variables) {
            if (OB_UNLIKELY(NULL == (plan_cache = plan_cache_))) {
              ret = OB_ERR_UNEXPECTED;
            } else {
              plan_cache->inc_access_cnt();
              if (OB_SQL_PC_NOT_EXIST == get_plan_err) {
                add_plan_to_pc = true;
              } else {
                add_plan_to_pc = false;
              }
            }
          }
          if (OB_SUCC(ret) && stmt::T_EXPLAIN == stmt_type) {
            if (OB_SQL_PC_NOT_EXIST == get_plan_err) {
              is_explain_parameterize = true;
            } else {
              is_explain_parameterize = false;
            }
          }
        }
      }
    }
  }

  if (OB_SUCC(ret)) {
    // Database runtime read-only check.
    if ((session->is_inner() && !session->is_user_session()) || pc_ctx.is_begin_commit_stmt()) {
      // Internal SQL and transaction control are not governed by database read-only state.
    } else if (OB_ISNULL(pc_ctx.sql_ctx_.schema_guard_)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(check_read_only_privilege(parse_result, exec_ctx,
        *pc_ctx.sql_ctx_.schema_guard_, pc_ctx.sql_traits_))) {
    }
  }

  if (OB_FAIL(ret)) {
     // do nothing
  } else if (!is_enable_transform_tree) {
    //pc_ctx.fp_result_.pc_key_.name_
    if (OB_FAIL(ob_write_string(allocator,
                                pc_ctx.raw_sql_,
                                pc_ctx.fp_result_.pc_key_.name_))) {
    } else if (OB_FAIL(ob_write_string(allocator,
                                       pc_ctx.raw_sql_,
                                       pc_ctx.sql_ctx_.plan_key_.constructed_sql_))) {
    }
  } else {
    // Outline statements also need syntax-tree transformation so parameter markers in
    // their embedded SQL are validated consistently.
    bool flag = false;
    if ((add_plan_to_pc && !is_show_variables) ||
        is_explain_parameterize ||
        ((T_CREATE_OUTLINE == parse_result.result_tree_->children_[0]->type_ ||
          T_ALTER_OUTLINE == parse_result.result_tree_->children_[0]->type_) &&
         (INT64_MAX != parse_result.result_tree_->children_[0]->value_))) {
      flag = true;
      if (T_CREATE_OUTLINE == parse_result.result_tree_->children_[0]->type_) {
        if (1 != parse_result.result_tree_->children_[0]->children_[2]->value_) {
          flag = false;
        }
      }
    } else if (pc_ctx.sql_ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
      if (T_EXPLAIN == parse_result.result_tree_->children_[0]->type_) {
        flag = true;
      }
    }
    if (flag) {
      bool is_transform_outline = (T_CREATE_OUTLINE == parse_result.result_tree_->children_[0]->type_
                                   || T_ALTER_OUTLINE == parse_result.result_tree_->children_[0]->type_);
      if (is_transform_outline) {
        LOG_WARN("is_transform_outline", K(is_transform_outline));
      }
      if (OB_FAIL(ObSqlParameterization::parameterize_syntax_tree(allocator,
                                                                  is_transform_outline,
                                                                  pc_ctx,
                                                                  parse_result.result_tree_,
                                                                  pctx->get_param_store_for_update(),
                                                                  session->get_charsets4parser()))) {
        bool need_retry_param = true;
        int tmp_ret = OB_SUCCESS;
        tmp_ret = OB_E(EventTable::EN_SQL_PARAM_FP_NP_NOT_SAME_ERROR) OB_SUCCESS;
        if (OB_SUCCESS != tmp_ret) {
          if (OB_NOT_SUPPORTED == ret) {
            need_retry_param = false;
          }
        }
        if (!need_retry_param) {
          // do nothing
        } else if (is_transform_outline) {
        } else {
          // If it is due to parameterization error, then a new parser needs to be performed, generating a new parser tree, the previous parser tree may have been partially parameterized,
          // and mark this query not to go into plan cache, and no need for parameterization next time, thus ensuring that errors during parameterization do not affect normal execution.
          pctx->reset_datum_param_store();
          is_enable_transform_tree = false;
          if (OB_FAIL(SMART_CALL(parser_and_check(outlined_stmt,
                                                  exec_ctx,
                                                  pc_ctx,
                                                  parse_result,
                                                  get_plan_err,
                                                  add_plan_to_pc,
                                                  is_enable_transform_tree)))) {
          }
          add_plan_to_pc = false;
        }
      } else {
        parse_result.question_mark_ctx_.count_ = static_cast<int> (pctx->get_param_store().count());
      }
    }
  }

  return ret;
}

int ObSql::pc_add_plan(ObPlanCacheCtx &pc_ctx,
                       ObResultSet &result,
                       ObOutlineState &outline_state,
                       ObPlanCache *plan_cache,
                       bool& plan_added)
{
  int ret = OB_SUCCESS;
  ObPhysicalPlan *phy_plan = result.get_physical_plan();
  pc_ctx.fp_result_.pc_key_.namespace_ = ObLibCacheNameSpace::NS_CRSR;
  plan_added = false;
  bool is_batch_exec = pc_ctx.sql_ctx_.is_batch_params_execute();
  if (OB_ISNULL(phy_plan) || OB_ISNULL(plan_cache)) {
    ret = OB_NOT_INIT;
    LOG_WARN("Fail to generate plan", K(phy_plan), K(plan_cache));
  } else if (OB_USE_PLAN_CACHE_NONE == phy_plan->get_phy_plan_hint().plan_cache_policy_) {
    if (is_batch_exec) {
      ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
      LOG_WARN("with not use plan_cache hint, batched multi_stmt needs rollback", K(ret));
    }
  } else if (OB_FAIL(result.to_plan(pc_ctx.mode_, phy_plan))) {
  } else if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                     (PC_PS_MODE == pc_ctx.mode_ || PC_PL_MODE == pc_ctx.mode_)
                                       ? pc_ctx.raw_sql_ :
                                       pc_ctx.sql_ctx_.plan_key_.constructed_sql_,
                                     phy_plan->stat_.constructed_sql_))) {
  } else if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                     pc_ctx.sql_ctx_.plan_key_.sql_id_,
                                     phy_plan->stat_.sql_id_))) {
  } else if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                     pc_ctx.sql_ctx_.plan_key_.format_sql_id_,
                                     phy_plan->stat_.format_sql_id_))) {
  } else {
    phy_plan->set_outline_state(outline_state);
    phy_plan->stat_.db_id_ = pc_ctx.sql_ctx_.plan_key_.db_id_;
    phy_plan->stat_.is_inner_ = result.get_session().is_inner();

    if (PC_PS_MODE == pc_ctx.mode_ || PC_PL_MODE == pc_ctx.mode_) {
      // pc_key_ may be modified elsewhere, so reset it before adding plan
      pc_ctx.fp_result_.pc_key_.key_id_ = pc_ctx.sql_ctx_.statement_id_;
      ret = plan_cache->add_ps_plan(phy_plan, pc_ctx);
    } else {
      ret = plan_cache->add_plan(phy_plan, pc_ctx);
    }
    plan_added = (OB_SUCCESS == ret);
    int tmp_ret = OB_SUCCESS;
    tmp_ret = OB_E(EventTable::EN_PC_NOT_SWALLOW_ERROR) OB_SUCCESS;
    if (is_batch_exec) {
      // Batch optimization cannot continue for errors other than OB_SQL_PC_PLAN_DUPLICATE.
      if (OB_SQL_PC_PLAN_DUPLICATE == ret) {
        ret = OB_SUCCESS;
      } else if (OB_FAIL(ret)) {
        ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
      } else {
        pc_ctx.sql_ctx_.self_add_plan_ = true;
      }
    } else if (OB_SQL_PC_PLAN_DUPLICATE == ret) {
      ret = OB_SUCCESS;
    } else if (OB_REACH_MEMORY_LIMIT == ret || OB_SQL_PC_PLAN_SIZE_LIMIT == ret) {
      if (REACH_TIME_INTERVAL(1000000)) { //1s, when memory reaches its limit, this log print will be relatively frequent, so it prints at an interval of 1s
        ObTruncatedString trunc_sql(pc_ctx.raw_sql_);
        LOG_INFO("can't add plan to plan cache",
                 K(ret), K(phy_plan->get_mem_size()), K(trunc_sql),
                 K(plan_cache->get_mem_used()));
      }
      ret = OB_SUCCESS;
    } else if (is_not_supported_err(ret)) {
      ret = OB_SUCCESS;
    } else if (OB_FAIL(ret)) {
      if (OB_SUCCESS != tmp_ret) {

      } else {
        ret = OB_SUCCESS; // add plan error, overwrite error code, ensure that failure of plan cache does not affect normal execution path
      }
    } else {
      pc_ctx.sql_ctx_.self_add_plan_ = true;
    }
  }

  return ret;
}
int ObSql::after_get_plan(ObPlanCacheCtx &pc_ctx,
                          ObSQLSessionInfo &session,
                          ObPhysicalPlan *phy_plan,
                          bool from_plan_cache)
{
  int ret = OB_SUCCESS;
  ObPhysicalPlanCtx *pctx = pc_ctx.exec_ctx_.get_physical_plan_ctx();
  if (NULL != pctx) {
    if (NULL != phy_plan) {
      // record the plan id in trace_event, perf_event and atomic_event
      NG_TRACE_EXT(plan_id, OB_ID(plan_id), phy_plan->get_plan_id());
      if (OB_MAX_SQL_ID_LENGTH != phy_plan->stat_.sql_id_.length()) {
        if (OB_FAIL(ob_write_string(phy_plan->get_allocator(),
                                    pc_ctx.sql_ctx_.sql_id_,
                                    phy_plan->stat_.sql_id_))) {
        }
      }
      // init auto increment param
      if (OB_FAIL(ret)) {
      } else if (OB_FAIL(pc_ctx.exec_ctx_.init_physical_plan_ctx(*phy_plan))) {
      } else if (OB_FAIL(DAS_CTX(pc_ctx.exec_ctx_).init(*phy_plan, pc_ctx.exec_ctx_))) {
      } else if (OB_FAIL(pctx->set_autoinc_params(phy_plan->get_autoinc_params()))) {
      } else {
        pctx->set_tablet_autoinc_param(phy_plan->get_tablet_autoinc_param());
        ObIArray<AutoincParam> &autoinc_params = pctx->get_autoinc_params();
        for (int64_t i = 0; OB_SUCC(ret) && i < autoinc_params.count(); ++i) {
          AutoincParam &param = autoinc_params.at(i);
          // Since 4.0, there should be only one autoinc_param
          param.autoinc_increment_ = session.get_local_auto_increment_increment();
          param.autoinc_offset_ = session.get_local_auto_increment_offset();
          if (pc_ctx.sql_ctx_.is_do_insert_batch_opt()) {
            param.total_value_count_ = pc_ctx.sql_ctx_.get_insert_batch_row_cnt();
          } else if (phy_plan->is_plain_insert() && pctx->get_array_param_groups().count() == 1) {
            param.total_value_count_ = pctx->get_array_param_groups().at(0).row_count_;
          }
        }  // end for
      }

    }

    if (OB_SUCC(ret)) {
      // set last_insert_id
      pctx->set_last_insert_id_session(session.get_local_last_insert_id());
    }

    if (OB_SUCC(ret) && NULL != phy_plan) {
      if (from_plan_cache) {
        ObPhyPlanHint &phy_plan_hint = phy_plan->get_phy_plan_hint();
        pc_ctx.sql_ctx_.force_print_trace_ = phy_plan_hint.force_trace_log_;
        //log_level just deal mpquery now.
        if (MpQuery == pc_ctx.sql_ctx_.exec_type_ && phy_plan_hint.log_level_.length() > 0) {
          if (OB_FAIL(process_thread_log_id_level_map(phy_plan_hint.log_level_.ptr(),
                                                      phy_plan_hint.log_level_.length()))) {
          }
        }
      }
    }
    if (OB_SUCC(ret) && NULL != phy_plan) {
      if (OB_FAIL(pctx->set_all_local_session_vars(phy_plan->get_all_local_session_vars()))) {
      }
    }
  } else {
    // not phy_plan, ignore
  }

  return ret;
}

int ObSql::need_add_plan(const ObPlanCacheCtx &pc_ctx,
                         ObResultSet &result,
                         bool is_enable_pc,
                         bool &need_add_plan)
{
  int ret = OB_SUCCESS;
  result.get_exec_context().get_stmt_factory()->get_query_ctx();
  if (false == need_add_plan) {
    // do nothing
  } else if (!is_enable_pc || !pc_ctx.should_add_plan_) {
    need_add_plan = false;
  } else if (OB_ISNULL(result.get_exec_context().get_stmt_factory())
             || OB_ISNULL(result.get_exec_context().get_stmt_factory()->get_query_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (result.get_exec_context().get_stmt_factory()->get_query_ctx()->has_hybrid_search()) {
    need_add_plan = false;
  }
  return ret;
}

OB_NOINLINE int ObSql::handle_physical_plan(const ObString &trimed_stmt,
                                            ObSqlCtx &context,
                                            ObResultSet &result,
                                            ObPlanCacheCtx &pc_ctx,
                                            const int get_plan_err)
{
  int ret = OB_SUCCESS;
  bool is_valid = true;
  PlanCacheMode mode = pc_ctx.mode_;
  ObString outlined_stmt = trimed_stmt;//use outline if available
  ObString signature_sql;
  ObString signature_format_sql;
  ObOutlineState outline_state;
  ParseResult parse_result;
  MEMSET(&parse_result, 0, SIZEOF(ParseResult));
  ParseResult outline_parse_result;
  MEMSET(&outline_parse_result, 0, SIZEOF(ParseResult));
  bool add_plan_to_pc = false;
  ObSQLSessionInfo &session = result.get_session();
  ObPlanCache *plan_cache = plan_cache_;
  bool use_plan_cache = session.get_local_ob_enable_plan_cache();
  // record whether needs to do parameterization at this time,
  // if exact mode is on, not do parameterizaiton
  bool is_enable_transform_tree = !session.get_enable_exact_mode();
  // Reset these two flags before reparsing to avoid issues with non-idempotency of these parameters after regenerating the plan due to previous plan cache lookup operations
  pc_ctx.not_param_index_.reset();
  pc_ctx.neg_param_index_.reset();
  bool plan_added = false;
  // for batched multi stmt, we only parse and optimize the first statement
  // only in multi_query, need do this
  if (!(PC_PS_MODE == mode || PC_PL_MODE == mode) &&
      (context.is_batch_params_execute() || pc_ctx.exec_ctx_.has_dynamic_values_table()) &&
      OB_FAIL(get_reconstructed_batch_stmt(pc_ctx, outlined_stmt))) {
  } else if (OB_FAIL(handle_parser(outlined_stmt,
                                   result.get_exec_context(),
                                   pc_ctx,
                                   parse_result,
                                   get_plan_err,
                                   add_plan_to_pc,
                                   is_enable_transform_tree))) {
  } else if (context.is_batch_params_execute() &&
             !(PC_PS_MODE == mode || PC_PL_MODE == mode) &&
             OB_FAIL(check_batched_multi_stmt_after_parser(pc_ctx,
                                                           parse_result,
                                                           add_plan_to_pc,
                                                           is_valid))) {
  } else if (!is_valid) {
    ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
    LOG_WARN("batched multi_stmt needs rollback", K(ret));
  }
  generate_sql_id(pc_ctx, add_plan_to_pc, parse_result, signature_sql, signature_format_sql, ret);

  if (OB_FAIL(ret)) {
    // do nothing
  } else if (OB_FAIL(get_outline_data(context, pc_ctx, signature_sql, signature_format_sql,
                                      outline_state, outline_parse_result))) {
  } else if (OB_FAIL(generate_physical_plan(parse_result,
                                            &pc_ctx,
                                            context,
                                            result,
                                            pc_ctx.is_begin_commit_stmt(),
                                            mode,
                                            &outline_parse_result))) {
  } else if (OB_FAIL(need_add_plan(pc_ctx,
                                   result,
                                   use_plan_cache,
                                   add_plan_to_pc))) {
  } else if (!add_plan_to_pc) {
    // do nothing
  } else if (OB_FAIL(pc_add_plan(pc_ctx, result, outline_state, plan_cache, plan_added))) {
  }
  //if the error code is ob_timeout, we add more error info msg for dml query.
  if (OB_UNLIKELY(OB_TIMEOUT == ret && session.is_user_session())) {
    LOG_USER_ERROR(OB_TIMEOUT, THIS_WORKER.get_timeout_ts() - result.get_session().get_query_start_time());
  }
  return ret;
}

int ObSql::handle_parser(const ObString &sql,
                         ObExecContext &exec_ctx,
                         ObPlanCacheCtx &pc_ctx,
                         ParseResult &parse_result,
                         int get_plan_err,
                         bool &add_plan_to_pc,
                         bool &is_enable_transform_tree)

{
  int ret = OB_SUCCESS;
  int64_t last_mem_usage = pc_ctx.allocator_.total();
  int64_t parser_mem_usage = 0;
  ObPhysicalPlanCtx *pctx = exec_ctx.get_physical_plan_ctx();
  const ObSqlCtx *sql_ctx = exec_ctx.get_sql_ctx();
  if (OB_ISNULL(pc_ctx.sql_ctx_.session_info_) || OB_ISNULL(pctx)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(SMART_CALL(parser_and_check(sql, exec_ctx, pc_ctx, parse_result,
                                      get_plan_err, add_plan_to_pc, is_enable_transform_tree)))) {
  }
  if (OB_SUCC(ret)) {
    if (exec_ctx.has_dynamic_values_table()) {
      if (OB_FAIL(ObValuesTableCompression::resolve_params_for_values_clause(pc_ctx))) {
      }
    }
    if (OB_SUCC(ret)) {
      pctx->set_original_param_cnt(pctx->get_param_store().count());
    }
  }
  return ret;
}

int ObSql::check_batched_multi_stmt_after_parser(ObPlanCacheCtx &pc_ctx,
                                                 ParseResult &parse_result,
                                                 bool add_plan_to_pc,
                                                 bool &is_valid)
{
  int ret = OB_SUCCESS;
  is_valid = false;
  ObItemType type = parse_result.result_tree_->children_[0]->type_;
  if (add_plan_to_pc ||
      (ObSQLUtils::is_enable_explain_batched_multi_statement() && T_EXPLAIN == type)) {
    is_valid = true;
    // only update support batched multi-stmt optimization
    if (OB_ISNULL(parse_result.result_tree_) ||
        OB_ISNULL(parse_result.result_tree_->children_)) {
      ret = OB_ERR_UNEXPECTED;;
    } else if (OB_UNLIKELY(parse_result.result_tree_->num_child_ < 1)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_ISNULL(parse_result.result_tree_->children_[0])) {
      ret = OB_ERR_UNEXPECTED;
    } else if (!(ObSQLUtils::is_support_batch_exec(type) || T_EXPLAIN == type)) {
      is_valid = false;
    } else { /*do nothing*/ }

    if (OB_SUCC(ret) && is_valid && !pc_ctx.not_param_info_.empty()) {
      if (pc_ctx.sql_ctx_.is_do_insert_batch_opt()) {
        if (OB_FAIL(ObPlanCacheValue::check_insert_multi_values_param(pc_ctx, is_valid))) {
        }
      } else if (pc_ctx.sql_ctx_.multi_stmt_item_.is_batched_multi_stmt()) {
        if (OB_FAIL(ObPlanCacheValue::check_multi_stmt_not_param_value(pc_ctx.multi_stmt_fp_results_,
                                                                       pc_ctx.not_param_info_,
                                                                       is_valid))) {
        }
      }
    }
  }
  return ret;
}

int ObSql::before_resolve_array_params(ObPlanCacheCtx &pc_ctx,
                                       int64_t query_num,
                                       int64_t param_num,
                                       ParamStore *&ab_params,
                                       ObBitSet<> &neg_param_index,
                                       ObBitSet<> &not_param_index,
                                       ObBitSet<> &must_be_positive_index,
                                       ObBitSet<> &fmt_int_or_ch_decint_idx)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(ab_params = static_cast<ParamStore *>(pc_ctx.allocator_.alloc(sizeof(ParamStore))))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else if (FALSE_IT(ab_params = new(ab_params)ParamStore(ObWrapperAllocator(pc_ctx.allocator_)))) {
   // do nothing
  } else if (OB_FAIL(ObSQLUtils::create_multi_stmt_param_store(pc_ctx.allocator_,
                                                               query_num,
                                                               param_num,
                                                               *ab_params))) {
  } else if (OB_FAIL(neg_param_index.add_members2(pc_ctx.neg_param_index_))) {
  } else if (OB_FAIL(not_param_index.add_members2(pc_ctx.not_param_index_))) {
  } else if (OB_FAIL(must_be_positive_index.add_members2(pc_ctx.must_be_positive_index_))) {
  } else if (OB_FAIL(fmt_int_or_ch_decint_idx.add_members2(pc_ctx.fmt_int_or_ch_decint_idx_))) {
  }
  return ret;
}

int ObSql::resolve_ins_multi_row_params(ObPlanCacheCtx &pc_ctx, const ObStmt &stmt, ParamStore *&ab_params)
{
  int ret = OB_SUCCESS;
  ObPhysicalPlanCtx *plan_ctx = NULL;
  ObBitSet<> neg_param_index;
  ObBitSet<> not_param_index;
  ObBitSet<> must_be_positive_index;
  ObBitSet<> formalize_prec_idx;
  int64_t query_num = pc_ctx.sql_ctx_.get_insert_batch_row_cnt();
  int64_t param_num = 0;
  if (OB_ISNULL(plan_ctx = pc_ctx.exec_ctx_.get_physical_plan_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (FALSE_IT(param_num = plan_ctx->get_param_store().count())) {
    // do nothing
  } else if (OB_FAIL(before_resolve_array_params(pc_ctx,
                                                 query_num,
                                                 param_num,
                                                 ab_params,
                                                 neg_param_index,
                                                 not_param_index,
                                                 must_be_positive_index,
                                                 formalize_prec_idx))) {
  } else if (OB_FAIL(ObPlanCacheValue::resolve_insert_multi_values_param(pc_ctx,
                                                                         stmt.get_stmt_type(),
                                                                         pc_ctx.param_charset_type_,
                                                                         neg_param_index,
                                                                         not_param_index,
                                                                         must_be_positive_index,
                                                                         formalize_prec_idx,
                                                                         param_num,
                                                                         *ab_params))) {
  }
  return ret;
}

int ObSql::resolve_multi_query_params(ObPlanCacheCtx &pc_ctx, const ObStmt &stmt, ParamStore *&ab_params)
{
  int ret = OB_SUCCESS;
  ObPhysicalPlanCtx *plan_ctx = NULL;
  ObBitSet<> neg_param_index;
  ObBitSet<> not_param_index;
  ObBitSet<> must_be_positive_index;
  ObBitSet<> formalize_prec_index;
  int64_t query_num = pc_ctx.sql_ctx_.get_batch_params_count();
  int64_t param_num = 0;
  if (OB_ISNULL(plan_ctx = pc_ctx.exec_ctx_.get_physical_plan_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (FALSE_IT(param_num = plan_ctx->get_param_store().count())) {
    // do nothing
  } else if (OB_FAIL(before_resolve_array_params(pc_ctx,
                                                 query_num,
                                                 param_num,
                                                 ab_params,
                                                 neg_param_index,
                                                 not_param_index,
                                                 must_be_positive_index,
                                                 formalize_prec_index))) {
  } else if (OB_FAIL(ObPlanCacheValue::check_multi_stmt_param_type(pc_ctx,
                                                                   stmt.get_stmt_type(),
                                                                   pc_ctx.param_charset_type_,
                                                                   neg_param_index,
                                                                   not_param_index,
                                                                   must_be_positive_index,
                                                                   formalize_prec_index,
                                                                   *ab_params))) {
  }

  return ret;
}

int ObSql::check_batched_multi_stmt_after_resolver(ObPlanCacheCtx &pc_ctx,
                                                   const ObStmt &stmt,
                                                   bool &is_valid)
{
  int ret = OB_SUCCESS;
  ObPhysicalPlanCtx *plan_ctx = NULL;
  is_valid = true;
  if (OB_ISNULL(plan_ctx = pc_ctx.exec_ctx_.get_physical_plan_ctx())
      || OB_ISNULL(pc_ctx.sql_ctx_.session_info_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (!(stmt.is_support_batch_exec_stmt() || stmt.is_explain_stmt())) {
    is_valid = false;
  } else {
    const ObDelUpdStmt &delupd_stmt = stmt.is_explain_stmt()
        ? static_cast<const ObDelUpdStmt&>(*(static_cast<const ObExplainStmt&>(stmt).get_explain_query_stmt()))
        : static_cast<const ObDelUpdStmt&>(stmt);
    if (delupd_stmt.is_update_stmt() || delupd_stmt.is_delete_stmt()) {
      if (1 != delupd_stmt.get_table_items().count() ||
          !delupd_stmt.get_table_items().at(0)->is_basic_table()) {
        is_valid = false;
      }
    }
    if (delupd_stmt.has_order_by() || delupd_stmt.has_limit()) {
      is_valid = false;
    }

    // make sure type of all the parameters are the same
    if (OB_SUCC(ret) && is_valid) {
      ParamStore *ab_params = NULL;
      ObBitSet<> neg_param_index;
      ObBitSet<> not_param_index;
      ObBitSet<> must_be_positive_index;
      if (pc_ctx.sql_ctx_.is_do_insert_batch_opt()) {
        if (OB_FAIL(resolve_ins_multi_row_params(pc_ctx, stmt, ab_params))) {
        } else {
          pc_ctx.ab_params_ = ab_params;
          ParamStore &param_store = plan_ctx->get_param_store_for_update();
          for (int64_t i = 0; OB_SUCC(ret) && i < param_store.count(); i++) {
            ObObjParam &obj_param = param_store.at(i);
            obj_param.get_param_flag().is_batch_parameter_ = true;
          }
        }
      } else if (!pc_ctx.sql_ctx_.multi_stmt_item_.is_ab_batch_opt()) {
        if (OB_FAIL(resolve_multi_query_params(pc_ctx, stmt, ab_params))) {
        } else {
          pc_ctx.ab_params_ = ab_params;
          ParamStore &param_store = plan_ctx->get_param_store_for_update();
          for (int64_t i = 0; OB_SUCC(ret) && i < param_store.count(); i++) {
            ObObjParam &obj_param = param_store.at(i);
            obj_param.get_param_flag().is_batch_parameter_ = true;
          }
        }
      }
    }
  }
  return ret;
}

void ObSql::generate_ps_sql_id(const ObString &raw_sql,
                               ObSqlCtx &context)
{
  (void)ObSQLUtils::md5(raw_sql, context.sql_id_, (int32_t)sizeof(context.sql_id_));
}

void ObSql::generate_sql_id(ObPlanCacheCtx &pc_ctx,
                           bool add_plan_to_pc,
                           ParseResult &parse_result,
                           ObString &signature_sql,
                           ObString &signature_format_sql,
                           int err_code)
{
  // It has been checked during parser_and_check, there is no need to check again here
  if (OB_SUCCESS == err_code
      && PC_TEXT_MODE == pc_ctx.mode_
      && T_SP_CALL_STMT == parse_result.result_tree_->children_[0]->type_) {
    signature_sql = pc_ctx.fp_result_.pc_key_.name_;
  } else if (add_plan_to_pc == false
            || PC_PS_MODE == pc_ctx.mode_
            || PC_PL_MODE == pc_ctx.mode_
            || OB_SUCCESS != err_code) {
    signature_sql = pc_ctx.raw_sql_;
   // if err happens in parameterization, not generate format_sql;
    signature_format_sql.reset();
  } else {
    signature_sql = pc_ctx.sql_ctx_.plan_key_.constructed_sql_;
    signature_format_sql = pc_ctx.sql_ctx_.plan_key_.format_sql_;
  }
  (void)ObSQLUtils::md5(signature_sql,
                        pc_ctx.sql_ctx_.sql_id_,
                        (int32_t)sizeof(pc_ctx.sql_ctx_.sql_id_));
  pc_ctx.sql_ctx_.plan_key_.sql_id_.
    assign_ptr(pc_ctx.sql_ctx_.sql_id_, static_cast<ObString::obstr_size_t>(strlen(pc_ctx.sql_ctx_.sql_id_)));

  if (!signature_format_sql.empty()) {
    (void)ObSQLUtils::md5(signature_format_sql,
                          pc_ctx.sql_ctx_.format_sql_id_,
                          (int32_t)sizeof(pc_ctx.sql_ctx_.format_sql_id_));
    pc_ctx.sql_ctx_.plan_key_.format_sql_id_.
      assign_ptr(pc_ctx.sql_ctx_.format_sql_id_, static_cast<ObString::obstr_size_t>(strlen(pc_ctx.sql_ctx_.format_sql_id_)));
  }
}


int ObSql::calc_pre_calculable_exprs(const ObDMLStmt &stmt,
                                 const ObIArray<ObHiddenColumnItem> &calculable_exprs,
                                 const bool is_ignore_stmt,
                                 ObExecContext &exec_ctx,
                                 ObPhysicalPlan &phy_plan,
                                 const uint64_t calc_types) /* default PRE_CALC_DEFAULT */
{
  int ret = OB_SUCCESS;
  OB_ASSERT(NULL != exec_ctx.get_physical_plan_ctx() &&
            NULL != exec_ctx.get_my_session() &&
            NULL != exec_ctx.get_stmt_factory() &&
            NULL != exec_ctx.get_stmt_factory()->get_query_ctx());
  ObPhysicalPlanCtx *phy_plan_ctx = exec_ctx.get_physical_plan_ctx();
  ObPreCalcExprFrameInfo *pre_calc_frame = NULL;
  void *frame_buf = NULL;
  bool need_fetch_cur_time = false;
  if (OB_ISNULL(phy_plan_ctx)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(frame_buf = phy_plan.get_allocator().alloc(
                                                            sizeof(ObPreCalcExprFrameInfo)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    phy_plan_ctx->set_ignore_stmt(is_ignore_stmt);
    DatumParamStore &datum_param_store = phy_plan_ctx->get_datum_param_store();
    ObStaticEngineExprCG expr_cg(phy_plan.get_allocator(),
                                 exec_ctx.get_my_session(),
                                 exec_ctx.get_sql_ctx()->schema_guard_,
                                 phy_plan_ctx->get_original_param_cnt(),
                                 datum_param_store.count());
    pre_calc_frame = new(frame_buf)ObPreCalcExprFrameInfo(phy_plan.get_allocator());
    if (OB_FAIL(expr_cg.generate_calculable_exprs(calculable_exprs,
                                                  *pre_calc_frame))) {
    } else {
      phy_plan.set_fetch_cur_time(stmt.get_fetch_cur_time());
      phy_plan.set_stmt_type(stmt.get_stmt_type());
      phy_plan.set_literal_stmt_type(exec_ctx.get_stmt_factory()->get_query_ctx()->get_literal_stmt_type());

      need_fetch_cur_time = phy_plan.get_fetch_cur_time() && !phy_plan_ctx->has_cur_time();
    }
    if (OB_FAIL(ret)) {
      // set current time before do pre calculation
    } else if (need_fetch_cur_time && FALSE_IT(phy_plan_ctx->set_cur_time(
                                  ObTimeUtility::current_time(), *(exec_ctx.get_my_session())))) {
      // do nothing
    } else if (OB_FAIL(ObPlanCacheObject::pre_calculation(is_ignore_stmt,
                                                          *pre_calc_frame, exec_ctx,
                                                          calc_types))) {
    } else if (OB_UNLIKELY(PRE_CALC_DEFAULT == calc_types &&
                   !phy_plan.get_pre_calc_frames().add_last(pre_calc_frame))) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      // do nothing
    }
  }
  return ret;
}

int ObSql::create_expr_constraints(ObQueryCtx &query_ctx, ObExecContext &exec_ctx)
{
  int ret = OB_SUCCESS;
  ObSEArray<ObHiddenColumnItem, 4> pre_calc_exprs;
  ObHiddenColumnItem hidden_column_item;
  int64_t idx = -1;
  if (query_ctx.all_expr_constraints_.empty()) {
    // do nothing
  } else {
    ObIArray<ObExprConstraint> &expr_constraints = query_ctx.all_expr_constraints_;
    ObSEArray<ObHiddenColumnItem, 4> pre_calc_exprs;
    ObHiddenColumnItem hidden_column_item;
    int64_t idx = -1;
    const int64_t dummy_count = -1;
    for (int64_t i = PRE_CALC_RESULT_NULL; OB_SUCC(ret) && i <= PRE_CALC_NOT_PRECISE; ++i) {
      PreCalcExprExpectResult expect_result = static_cast<PreCalcExprExpectResult>(i);
      pre_calc_exprs.reuse();
      for (int64_t j = 0; OB_SUCC(ret) && j < expr_constraints.count(); ++j) {
        if (expr_constraints.at(j).expect_result_ == expect_result) {
          hidden_column_item.expr_ = expr_constraints.at(j).pre_calc_expr_;
          hidden_column_item.hidden_idx_ = ++idx;
          if (OB_ISNULL(hidden_column_item.expr_)) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_FAIL(hidden_column_item.expr_->extract_info())) {
          } else if (!expr_constraints.at(j).ignore_const_check_ &&
                     OB_UNLIKELY(!ObOptEstUtils::is_calculable_expr(*hidden_column_item.expr_, dummy_count))) {
            ret = OB_ERR_UNEXPECTED;
          } else if (OB_FAIL(pre_calc_exprs.push_back(hidden_column_item))) {
          }
        }
      }
      if (OB_SUCC(ret) && !pre_calc_exprs.empty()) {
        if (OB_FAIL(create_expr_constraint(query_ctx,
                                           exec_ctx,
                                           pre_calc_exprs,
                                           expect_result))) {
        }
      }
    }
  }
  return ret;
}

int ObSql::create_expr_constraint(ObQueryCtx &query_ctx,
                                  ObExecContext &exec_ctx,
                                  const ObIArray<ObHiddenColumnItem> &pre_calc_exprs,
                                  const PreCalcExprExpectResult expect_result)
{
  int ret = OB_SUCCESS;
  ObPreCalcExprConstraint *pre_calc_constraint = NULL;
  void *cons_buf = NULL;
  ObPhysicalPlanCtx *plan_ctx = NULL;
  if (OB_ISNULL(exec_ctx.get_sql_ctx()) || OB_ISNULL(plan_ctx = exec_ctx.get_physical_plan_ctx())) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_ISNULL(cons_buf = exec_ctx.get_allocator().alloc(sizeof(ObPreCalcExprConstraint)))) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  } else {
    ObStaticEngineExprCG expr_cg(exec_ctx.get_allocator(), exec_ctx.get_my_session(),
                                 exec_ctx.get_sql_ctx()->schema_guard_,
                                 plan_ctx->get_original_param_cnt(),
                                 plan_ctx->get_datum_param_store().count());
    pre_calc_constraint = new(cons_buf)ObPreCalcExprConstraint(exec_ctx.get_allocator());
    pre_calc_constraint->expect_result_ = expect_result;
    if (OB_FAIL(expr_cg.generate_calculable_exprs(pre_calc_exprs,
                                                  pre_calc_constraint->pre_calc_expr_info_))) {
    } else if (OB_UNLIKELY(!query_ctx.all_pre_calc_constraints_.add_last(pre_calc_constraint))) {
      ret = OB_ERR_UNEXPECTED;
    }
  }
  return ret;
}

// handle execute in text protocol
int ObSql::handle_text_execute(const ObStmt *basic_stmt,
                               ObSqlCtx &sql_ctx,
                               ObResultSet &result)
{
  int ret = OB_SUCCESS;
  const ObExecuteStmt *exec_stmt = static_cast<const ObExecuteStmt*>(basic_stmt);
  sql_ctx.is_text_ps_mode_ = true;
  if (OB_ISNULL(exec_stmt)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    ObIAllocator &alloc = result.get_exec_context().get_allocator();
    ParamStore param_store((ObWrapperAllocator(alloc)));
    const ObRawExpr *raw_expr = NULL;
    const ObIArray<const ObRawExpr*> &raw_expr_params = exec_stmt->get_params();
    if (OB_FAIL(param_store.reserve(raw_expr_params.count()))) {
    }
    for (int64_t i = 0; OB_SUCC(ret) && i < raw_expr_params.count(); ++i) {
      ObObjParam obj_param;
      if (OB_ISNULL(raw_expr = raw_expr_params.at(i))) {
        ret = OB_ERR_UNEXPECTED;
      } else if (OB_FAIL(ObSQLUtils::calc_const_expr(result.get_exec_context(),
                                                     raw_expr,
                                                     obj_param,
                                                     alloc,
                                                     param_store))) {
      } else {
        obj_param.set_accuracy(raw_expr->get_accuracy());
        obj_param.set_result_flag(raw_expr->get_result_flag());
        obj_param.set_collation_level(CS_LEVEL_COERCIBLE);
        obj_param.set_param_meta(obj_param.get_meta());
        if (OB_FAIL(param_store.push_back(obj_param))) {
        }
      }
    }
    if (OB_SUCC(ret)) {
      if (OB_FAIL(handle_ps_execute(exec_stmt->get_prepare_id(),
                                    exec_stmt->get_prepare_type(),
                                    param_store,
                                    sql_ctx,
                                    result,
                                    false/*is_inner_sql*/))) {
      } else if (OB_FAIL(construct_param_store(param_store, result.get_ps_params()))) {
      }
    }
  }
  return ret;
}

int ObSql::get_reconstructed_batch_stmt(ObPlanCacheCtx &pc_ctx, ObString& stmt_sql)
{
  int ret = OB_SUCCESS;
  if (pc_ctx.sql_ctx_.is_do_insert_batch_opt()) {
    // Restore the original SQL according to the first row of parameters
    if (OB_FAIL(ObPlanCache::restore_param_to_truncated_sql(pc_ctx))) {
      // if rebuild origin sql fail, this sql would rollback
      ret = OB_BATCHED_MULTI_STMT_ROLLBACK;
    } else {
      stmt_sql = pc_ctx.insert_batch_opt_info_.new_reconstruct_sql_;
    }
  } else if (pc_ctx.sql_ctx_.handle_batched_multi_stmt()) {
    if (OB_ISNULL(pc_ctx.sql_ctx_.multi_stmt_item_.get_queries())) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_UNLIKELY(pc_ctx.sql_ctx_.multi_stmt_item_.get_queries()->empty())) {
      ret = OB_ERR_UNEXPECTED;
    } else {
      stmt_sql = pc_ctx.sql_ctx_.multi_stmt_item_.get_queries()->at(0);
    }
  } else if (pc_ctx.exec_ctx_.has_dynamic_values_table()) {
    stmt_sql = pc_ctx.new_raw_sql_;
  }
  return ret;
}

// If handel failed, rollback implicit started txn by current stmt
// if autocommit is on.
// this is because the txn may be started during generate plan for some
// strange Query.
// We have found one situation:
// udf contains DML stmt and with deterministic property in MySQL mode
void ObSql::rollback_implicit_trans_when_fail(ObResultSet &result, int &ret)
{
  bool ac = false;
  result.get_session().get_autocommit(ac);
  transaction::ObTxDesc *tx = result.get_session().get_tx_desc();
  if (ac && tx && data_plane::tx_desc_id(tx).is_valid()
      && !data_plane::tx_desc_is_explicit(tx)) {
    const transaction::ObTransID txid = data_plane::tx_desc_id(tx);
    int tmp_ret = OB_SUCCESS;
    bool need_disconnect = false;
    if (OB_TMP_FAIL(ObSqlTransControl::rollback_trans(&result.get_session(), need_disconnect))) {
      LOG_WARN("rollback transaction fail, will disconnect", K(tmp_ret), K(result.get_session()), K(txid));
      result.get_exec_context().set_need_disconnect(true);
      ret = tmp_ret;
    } else {
      LOG_INFO("rollback transaction started during get-plan success", K(ret), K(txid));
    }
  }
}


}//end of namespace sql
}//end of namespace oceanbase

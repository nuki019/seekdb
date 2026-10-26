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

#include "object/ob_object.h"
#define USING_LOG_PREFIX STORAGE_FTS

#include "share/rc/ob_server_runtime.h"
#include "storage/fts/ob_fts_stop_word.h"
#include "storage/fts/ob_fts_parser_helper.h"
#include "storage/fts/ob_fts_parser_property.h"

namespace oceanbase
{
namespace storage
{

////////////////////////////////////////////////////////////////////////////////
// class ObStopWordChecker
ObStopWordChecker::~ObStopWordChecker()
{
  destroy();
}

int ObStopWordChecker::init()
{
  int ret = OB_SUCCESS;
  
  

  if (inited_) {
    ret = OB_INIT_TWICE;
  } else if (OB_FAIL(stopword_set_.create(DEFAULT_STOPWORD_BUCKET_NUM, "StopWordSet", "StopWordSet"))) {
  } else {
    ObObjMeta stop_meta;
    stop_meta.set_varchar();
    stop_meta.set_collation_type(ObCollationType::CS_TYPE_UTF8MB4_GENERAL_CI);

    stopword_type_.set_meta(stop_meta);
    const int64_t stopword_count = sizeof(ob_stop_word_list) / sizeof(ob_stop_word_list[0]);
    for (int64_t i = 0; OB_SUCC(ret) && i < stopword_count; ++i) {
      ObFTWord stopword(STRLEN(ob_stop_word_list[i]), ob_stop_word_list[i], stopword_type_);
      if (OB_FAIL(stopword_set_.set_refactored(stopword))) {
      }
    }

    if (OB_SUCC(ret)) {
      inited_ = true;
    }
  }
  return ret;
}

void ObStopWordChecker::destroy()
{
  if (inited_) {
    stopword_set_.destroy();
    inited_ = false;
  }
}

int ObStopWordChecker::check_stopword(const ObFTWord &word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  
  
  if (OB_UNLIKELY(!inited_)) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    const common::ObString str = word.get_word().get_string();
    const char *ptr = str.ptr();
    int64_t len = str.length();
    const common::ObCollationType coll_type = word.get_collation_type();
    if (OB_LIKELY(common::ObCharset::charset_type_by_coll(coll_type) == common::CHARSET_UTF8MB4
                  && nullptr != ptr && len > 0)) {
      const uint8_t first_byte = static_cast<uint8_t>(ptr[0]);
      // 3-byte or 4-byte UTF-8 sequence (e.g., all CJK characters) can never match ASCII stopwords.
      if (first_byte >= 0xE0) {
        is_stopword = false;
        return OB_SUCCESS;
      }
      bool all_ascii = true;
      for (int64_t i = 0; i < len; ++i) {
        if (OB_UNLIKELY(static_cast<uint8_t>(ptr[i]) >= 0x80)) {
          all_ascii = false;
          break;
        }
      }
      if (OB_LIKELY(all_ascii)) {
        // UTF8MB4_GENERAL_CI uses PAD SPACE semantics; trim trailing ASCII spaces.
        while (len > 0 && ptr[len - 1] == ' ') {
          --len;
        }
        if (len < 1 || len > 5) {
          is_stopword = false;
          return OB_SUCCESS;
        }
        uint64_t packed = 0;
        for (int64_t i = 0; i < len; ++i) {
          uint8_t c = static_cast<uint8_t>(ptr[i]);
          if (c >= 'A' && c <= 'Z') {
            c |= 0x20;
          }
          packed = (packed << 8) | c;
        }
        #define PACK2(a, b) ((uint64_t(a) << 8) | uint64_t(b))
        #define PACK3(a, b, c) ((uint64_t(a) << 16) | (uint64_t(b) << 8) | uint64_t(c))
        #define PACK4(a, b, c, d) ((uint64_t(a) << 24) | (uint64_t(b) << 16) | (uint64_t(c) << 8) | uint64_t(d))
        #define PACK5(a, b, c, d, e) ((uint64_t(a) << 32) | (uint64_t(b) << 24) | (uint64_t(c) << 16) | (uint64_t(d) << 8) | uint64_t(e))
        switch (len) {
          case 1:
            is_stopword = (packed == 'a' || packed == 'i');
            break;
          case 2:
            switch (packed) {
              case PACK2('a','n'):
              case PACK2('a','s'):
              case PACK2('a','t'):
              case PACK2('b','e'):
              case PACK2('b','y'):
              case PACK2('d','e'):
              case PACK2('e','n'):
              case PACK2('i','n'):
              case PACK2('i','s'):
              case PACK2('i','t'):
              case PACK2('l','a'):
              case PACK2('o','f'):
              case PACK2('o','n'):
              case PACK2('o','r'):
              case PACK2('t','o'):
                is_stopword = true;
                break;
              default:
                is_stopword = false;
                break;
            }
            break;
          case 3:
            switch (packed) {
              case PACK3('a','r','e'):
              case PACK3('c','o','m'):
              case PACK3('f','o','r'):
              case PACK3('h','o','w'):
              case PACK3('t','h','e'):
              case PACK3('u','n','d'):
              case PACK3('w','a','s'):
              case PACK3('w','h','o'):
              case PACK3('w','w','w'):
                is_stopword = true;
                break;
              default:
                is_stopword = false;
                break;
            }
            break;
          case 4:
            switch (packed) {
              case PACK4('f','r','o','m'):
              case PACK4('t','h','a','t'):
              case PACK4('t','h','i','s'):
              case PACK4('w','h','a','t'):
              case PACK4('w','h','e','n'):
              case PACK4('w','i','l','l'):
              case PACK4('w','i','t','h'):
                is_stopword = true;
                break;
              default:
                is_stopword = false;
                break;
            }
            break;
          case 5:
            is_stopword = (packed == PACK5('a','b','o','u','t') || packed == PACK5('w','h','e','r','e'));
            break;
          default:
            is_stopword = false;
            break;
        }
        #undef PACK2
        #undef PACK3
        #undef PACK4
        #undef PACK5
        return OB_SUCCESS;
      }
    }
    common::ObArenaAllocator allocator(lib::ObMemAttr("ChkStopWord"));
    common::ObString cmp_str;
    // do nothing set out with in if type is the same.
    if (OB_FAIL(common::ObCharset::charset_convert(
                                       allocator,
                                       str,
                                       coll_type,
                                       stopword_type_.get_collation_type(),
                                       cmp_str))) {
    } else {
      ObFTWord converted(cmp_str.length(), cmp_str.ptr(), stopword_type_);
      ret = stopword_set_.exist_refactored(converted);
      if (OB_HASH_NOT_EXIST == ret) {
        is_stopword = false;
        ret = OB_SUCCESS;
      } else if (OB_HASH_EXIST == ret) {
        is_stopword = true;
        ret = OB_SUCCESS;
      } else if (OB_SUCC(ret)) {
        ret = OB_ERR_UNEXPECTED;
      } else {
      }
    }
  }
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
// class ObAddWord
ObAddWord::ObAddWord(
    const ObFTParserProperty &property,
    const ObObjMeta &meta,
    const ObAddWordFlag &flag,
    common::ObIAllocator &allocator,
    ObFTWordMap &word_map)
  : word_meta_(meta),
    allocator_(allocator),
    word_map_(word_map),
    min_max_word_cnt_(0),
    non_stopword_cnt_(0),
    stopword_cnt_(0),
    min_token_size_(property.min_token_size_),
    max_token_size_(property.max_token_size_),
    flag_(flag)
{
}

int ObAddWord::process_word(
    const char *word,
    const int64_t word_len,
    const int64_t char_cnt,
    const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  bool is_stopword = false;
  if (OB_ISNULL(word) || OB_UNLIKELY(0 >= word_len || 0 >= char_cnt || 0 >= word_freq)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (is_min_max_word(char_cnt)) {
    ++min_max_word_cnt_;
  } else {
    char stack_buf[256];
    const char *norm_ptr = word;
    int64_t norm_len = word_len;
    bool allocated_in_arena = false;
    if (flag_.casedown()) {
      const common::ObCollationType coll_type = word_meta_.get_collation_type();
      bool handled_fast = false;
      if (OB_LIKELY(common::ObCharset::charset_type_by_coll(coll_type) == common::CHARSET_UTF8MB4
                    && word_len <= static_cast<int64_t>(sizeof(stack_buf)))) {
        handled_fast = true;
        bool has_upper = false;
        int64_t i = 0;
        while (i < word_len) {
          const uint8_t b = static_cast<uint8_t>(word[i]);
          if (b < 0x80) {
            if (b >= 'A' && b <= 'Z') {
              has_upper = true;
            }
            ++i;
          } else if (b >= 0xE4 && b <= 0xE9 && i + 2 < word_len
                     && (static_cast<uint8_t>(word[i + 1]) & 0xC0) == 0x80
                     && (static_cast<uint8_t>(word[i + 2]) & 0xC0) == 0x80) {
            // Standard 3-byte CJK ideograph (U+4000..U+9FFF): tolower is identity
            i += 3;
          } else {
            handled_fast = false;
            break;
          }
        }
        if (handled_fast && has_upper) {
          for (int64_t j = 0; j < word_len; ++j) {
            const uint8_t c = static_cast<uint8_t>(word[j]);
            stack_buf[j] = static_cast<char>((c >= 'A' && c <= 'Z') ? (c | 0x20) : c);
          }
          norm_ptr = stack_buf;
        }
      }
      if (!handled_fast) {
        ObString src_str(word_len, word);
        ObString dst_str;
        if (OB_FAIL(ObCharset::tolower(coll_type, src_str, dst_str, allocator_))) {
          return ret;
        }
        norm_ptr = dst_str.ptr();
        norm_len = dst_str.length();
        allocated_in_arena = true;
      }
    }
    ObFTWord dst_word(norm_len, norm_ptr, word_meta_);
    if (OB_FAIL(check_stopword(dst_word, is_stopword))) {
    } else if (OB_UNLIKELY(is_stopword)) {
      ++stopword_cnt_;
    } else {
      if (flag_.groupby_word()) {
        int64_t *cnt_ptr = word_map_.get(dst_word);
        if (nullptr != cnt_ptr) {
          *cnt_ptr += word_freq;
          non_stopword_cnt_ += word_freq;
          return OB_SUCCESS;
        }
      }
      if (!allocated_in_arena) {
        char *buf = static_cast<char *>(allocator_.alloc(norm_len));
        if (OB_ISNULL(buf)) {
          return OB_ALLOCATE_MEMORY_FAILED;
        }
        MEMCPY(buf, norm_ptr, norm_len);
        dst_word = ObFTWord(norm_len, buf, word_meta_);
      }
      if (OB_FAIL(word_map_.set_refactored(dst_word, 1, flag_.groupby_word() ? 1 : 0))) {
      } else {
        non_stopword_cnt_ += word_freq;
      }
    }
  }
  return ret;
}

bool ObAddWord::is_min_max_word(const int64_t c_len) const
{
  return flag_.min_max_word() && (c_len < min_token_size_ || c_len > max_token_size_);
}

int ObAddWord::casedown_word(const ObFTWord &src, ObFTWord &dst)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(src.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.casedown()) {
    ObString dst_str;
    if (OB_FAIL(ObCharset::tolower(
                    word_meta_.get_collation_type(),
                    src.get_word().get_string(),
                    dst_str,
                    allocator_))) {
    } else {
      ObFTWord tmp(dst_str.length(), dst_str.ptr(), word_meta_);
      dst = tmp;
    }
  } else {
    dst = src;
  }
  return ret;
}

int ObAddWord::check_stopword(const ObFTWord &ft_word, bool &is_stopword)
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(ft_word.empty())) {
    ret = OB_INVALID_ARGUMENT;
  } else if (flag_.stopword()) {
    ObStopWordChecker *stop_word_checker = ObFTParseData::instance().stop_word_checker();
    if (OB_ISNULL(stop_word_checker)) {
      ret = OB_ERR_UNEXPECTED;
    } else if (OB_FAIL(stop_word_checker->check_stopword(ft_word, is_stopword))) {
    }
  }
  return ret;
}

int ObAddWord::groupby_word(const ObFTWord &word, const int64_t word_freq)
{
  int ret = OB_SUCCESS;
  int64_t word_count = 0;
  if (OB_UNLIKELY(word.empty() || word_freq <= 0)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (!flag_.groupby_word()) {
    if (OB_FAIL(word_map_.set_refactored(word, 1/*word count*/))) {
    }
  } else if (OB_FAIL(word_map_.get_refactored(word, word_count)) && OB_HASH_NOT_EXIST != ret) {
  } else {
    if (OB_HASH_NOT_EXIST == ret) {
      word_count = 1;
    } else {
      word_count += word_freq;
    }
    if (OB_FAIL(word_map_.set_refactored(word, word_count, 1/*overwrite*/))) {
    }
  }
  return ret;
}

} // end namespace storage
} // end namespace oceanbase

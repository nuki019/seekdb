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

#ifndef _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_
#define _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_

#include "lib/charset/ob_charset.h"

#include <cstdint>

namespace oceanbase
{
namespace storage
{
typedef int32_t ObFTWordCode;
typedef int32_t ObFTWordBase;
typedef uint32_t ObFTWordStateIndex;

/**
 * @class ObFTSingleWord
 * @brief sturct to store a single character of a charset;
 */
struct ObFTSingleWord
{
public:
  ObFTSingleWord() : word(""), word_len(0) {}
  ObFTSingleWord(const ObFTSingleWord &other) = default;
  ObFTSingleWord &operator=(const ObFTSingleWord &other) = default;

  OB_INLINE int32_t set_word(const char *w, int32_t len)
  {
    int ret = common::OB_SUCCESS;
    if (OB_ISNULL(w)) {
      ret = common::OB_INVALID_ARGUMENT;
    } else if (OB_UNLIKELY(len > common::ObCharset::MAX_MB_LEN)) {
      ret = common::OB_SIZE_OVERFLOW;
    } else {
      for (int32_t i = 0; i < len; ++i) {
        this->word[i] = w[i];
      }
      this->word_len = static_cast<uint8_t>(len);
    }
    return ret;
  }
  OB_INLINE common::ObString get_word() const { return common::ObString(word_len, word); }
  OB_INLINE bool operator==(const ObFTSingleWord &other) const
  {
    if (word_len != other.word_len) {
      return false;
    }
    switch (word_len) {
      case 3:
        return word[0] == other.word[0] && word[1] == other.word[1] && word[2] == other.word[2];
      case 1:
        return word[0] == other.word[0];
      case 2:
        return word[0] == other.word[0] && word[1] == other.word[1];
      case 4:
        return word[0] == other.word[0] && word[1] == other.word[1]
               && word[2] == other.word[2] && word[3] == other.word[3];
      default:
        return 0 == MEMCMP(word, other.word, word_len);
    }
  }

public:
  char word[common::ObCharset::MAX_MB_LEN];
  uint8_t word_len;
} __attribute__((packed));

enum class ObFTDictType : uint32_t
{
  DICT_TYPE_INVALID = 0,
  DICT_IK_MAIN = 1,
  DICT_IK_QUAN = 2,
  DICT_IK_STOP = 3,
};

class ObFTDictDesc
{
public:
  ObFTDictDesc(const ObString &name,
               const ObFTDictType type,
               const ObCharsetType charset,
               const ObCollationType coll_type)
      : name_(name), type_(type), charset_(charset), coll_type_(coll_type)
  {
  }

public:
  ObString name_;
  ObFTDictType type_;
  ObCharsetType charset_;
  ObCollationType coll_type_;
};

} //  namespace storage
} //  namespace oceanbase

#endif // _OCEANBASE_STORAGE_FTS_DICT_OB_FT_DICT_DEF_H_

/**
 * @file axis.hpp
 * @brief 基础类型别名：索引、轴编号等
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace nd {

/// 无符号索引类型（shape 维度长度、元素下标等）
using index_t  = std::size_t;

/// 有符号索引类型（stride、偏移量、负下标等）
using sindex_t = std::ptrdiff_t;

/// 轴编号类型
using axis_t   = std::size_t;

} // namespace nd

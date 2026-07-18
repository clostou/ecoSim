/**
 * @file ndarray.hpp  (include/ndarray/ndarray.hpp)
 * @brief ndarray 库统一入口头文件
 *
 * 使用方式：
 *   #include "ndarray/ndarray.hpp"
 *
 * 即可获得 nd::ndarray、nd::ndview 以及全部当前公开模块。
 *
 * @details
 * 当前入口按层聚合：
 *   1. 核心元数据层：shape / stride / layout / traits / utility
 *   2. 存储与视图层：allocator / buffer / ndarray / ndview
 *   3. 索引与形状变换层：slice / reshape / gather
 *   4. 逐元素表达式层：expression / unary / binary / math / transform
 *   5. 缩减层：reduce
 *   6. 收缩层：einsum labels / contract / named wrappers
 */

#pragma once

// Core
#include "core/axis.hpp"
#include "core/shape.hpp"
#include "core/stride.hpp"
#include "core/layout.hpp"
#include "core/broadcast.hpp"
#include "core/traits.hpp"
#include "core/utility.hpp"

// Storage
#include "storage/allocator.hpp"
#include "storage/buffer.hpp"

// Containers & Views
#include "view/ndview.hpp"
#include "view/ndarray.hpp"

// Indexing & Shape Transforms
#include "indexing/slice.hpp"
#include "indexing/reshape.hpp"
#include "indexing/gather.hpp"

// Elementwise Ops
#include "ops/expression.hpp"
#include "ops/unary.hpp"
#include "ops/binary.hpp"
#include "ops/math.hpp"
#include "ops/transform.hpp"

// Reduce
#include "reduce/reduce.hpp"

// Einsum
#include "einsum/labels.hpp"
#include "einsum/contract.hpp"
#include "einsum/einsum.hpp"

/**
 * @file broadcast.hpp
 * @brief NumPy 风格广播规则：形状兼容判定、广播形状推导与广播索引映射
 *
 * @details
 * 广播规则（与 NumPy 一致）：
 * 1. 从尾轴开始逐轴对齐比较。
 * 2. 较短形状在前端（左侧）隐式补 1。
 * 3. 对齐后，每一对轴要么相等，要么其中之一为 1。
 * 4. 广播输出在每个轴取两者中较大的 extent。
 *
 * 本头文件提供：
 * - `broadcast_shape<R>(a, b)`: 从两个 shape 推导广播输出形状
 * - `broadcastable<R>(a, b)`: 判断两个 shape 是否可广播
 * - `broadcast_index<Rank>`: 将输出坐标映射为源坐标的辅助
 */

#pragma once

#include "axis.hpp"
#include "shape.hpp"

#include <cassert>

namespace nd {

/**
 * @brief 判断两个同秩形状是否可广播
 *
 * @details
 * 同秩情况下，逐轴检查是否相等或其中之一为 1。
 * 不同秩的输入应先在模板层面通过左侧补 1 对齐到同秩再调用。
 */
template <std::size_t Rank>
constexpr bool broadcastable(const shape<Rank>& a, const shape<Rank>& b) {
    for (std::size_t i = 0; i < Rank; ++i) {
        if (a[i] != b[i] && a[i] != 1 && b[i] != 1) return false;
    }
    return true;
}

/**
 * @brief 计算两个同秩形状的广播输出形状
 *
 * @details
 * 前置条件：`broadcastable(a, b)` 为 true。
 * 输出在每个轴取 `max(a[i], b[i])`。
 */
template <std::size_t Rank>
constexpr shape<Rank> broadcast_shape(const shape<Rank>& a, const shape<Rank>& b) {
    assert(broadcastable(a, b) && "shapes are not broadcastable");
    shape<Rank> out;
    for (std::size_t i = 0; i < Rank; ++i) {
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
    return out;
}

// ================================================================
//  左侧补 1 对齐（不同秩情况）
// ================================================================

/**
 * @brief 将低秩形状左侧补 1 扩展到高秩
 *
 * @tparam OutRank 目标秩
 * @tparam InRank  输入秩（InRank <= OutRank）
 *
 * @details
 * 例如 shape<2>{3,4} → shape<4>{1,1,3,4}。
 */
template <std::size_t OutRank, std::size_t InRank>
constexpr shape<OutRank> pad_shape_left(const shape<InRank>& in) {
    static_assert(InRank <= OutRank, "InRank must not exceed OutRank");
    shape<OutRank> out;
    constexpr std::size_t pad = OutRank - InRank;
    for (std::size_t i = 0; i < pad; ++i) out[i] = 1;
    for (std::size_t i = 0; i < InRank; ++i) out[pad + i] = in[i];
    return out;
}

} // namespace nd

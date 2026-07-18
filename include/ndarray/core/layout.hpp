/**
 * @file layout.hpp
 * @brief 内存布局策略（row-major 等）
 */

#pragma once

#include "shape.hpp"
#include "stride.hpp"

namespace nd {

/**
 * @brief 行主序布局策略（C-order）
 *
 * @details
 * 最后一个轴变化最快，前一轴的 stride 等于后一轴 stride 与后一轴长度的乘积。
 * 这是当前库的默认布局，也是连续存储与大多数快路径的基准形式。
 */
struct row_major {
    /**
     * @brief 由 shape 推导行主序 stride
     * @tparam Rank 维度数
     * @param sh 输入 shape
     * @return 对应的连续行主序 stride
     */
    template <std::size_t Rank>
    static constexpr strides<Rank> compute_strides(const shape<Rank>& sh) {
        strides<Rank> st;
        st[Rank - 1] = 1;
        if constexpr (Rank > 1) {
            for (std::size_t i = Rank - 1; i > 0; --i) {
                st[i - 1] = st[i] * static_cast<sindex_t>(sh[i]);
            }
        }
        return st;
    }
};

/// @brief 默认布局别名，当前等价于 `row_major`
using row_major_layout = row_major;

} // namespace nd

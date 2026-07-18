/**
 * @file stride.hpp
 * @brief 固定秩 strides 类型，描述多维数组各轴的步长（元素粒度，有符号）
 */

#pragma once

#include "axis.hpp"

#include <array>
#include <type_traits>

namespace nd {

/**
 * @brief 固定秩步长对象
 *
 * @tparam Rank 维度数，必须在编译期确定
 *
 * @details
 * `strides<Rank>` 用“元素个数”而不是“字节数”表示各轴步长，因此可以直接
 * 与线性偏移公式配合使用。其值允许为负数，以支持转置、反向切片等非平凡视图。
 */
template <std::size_t Rank>
class strides {
    static_assert(Rank > 0, "Rank must be at least 1");

    std::array<sindex_t, Rank> data_{};

public:
    static constexpr std::size_t rank = Rank;

    // ---- 构造 ----

    /// @brief 默认构造为全 0 步长
    constexpr strides() = default;

    /// @brief 由 `std::array` 显式构造
    constexpr strides(const std::array<sindex_t, Rank>& d) : data_(d) {}

    /// @brief 由变参步长构造
    template <typename... Args,
              std::enable_if_t<sizeof...(Args) == Rank &&
                               std::conjunction_v<std::is_convertible<Args, sindex_t>...>, int> = 0>
    constexpr strides(Args... args) : data_{static_cast<sindex_t>(args)...} {}

    // ---- 元素访问 ----

    /// @brief 读取第 i 个轴的 stride
    constexpr sindex_t  operator[](std::size_t i) const { return data_[i]; }

    /// @brief 读写第 i 个轴的 stride
    constexpr sindex_t& operator[](std::size_t i)       { return data_[i]; }

    /// @brief 以 `std::array` 形式访问底层存储
    constexpr const std::array<sindex_t, Rank>& raw() const { return data_; }

    /// @brief 以 `std::array` 形式访问底层存储（可写）
    constexpr       std::array<sindex_t, Rank>& raw()       { return data_; }

    // ---- 迭代 ----

    constexpr auto begin() const { return data_.begin(); }
    constexpr auto end()   const { return data_.end();   }
    constexpr auto begin()       { return data_.begin(); }
    constexpr auto end()         { return data_.end();   }

    // ---- 比较 ----

    /// @brief 比较两个 stride 描述是否完全一致
    constexpr bool operator==(const strides& o) const { return data_ == o.data_; }

    /// @brief 比较两个 stride 描述是否不同
    constexpr bool operator!=(const strides& o) const { return data_ != o.data_; }
};

} // namespace nd

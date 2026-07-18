/**
 * @file shape.hpp
 * @brief 固定秩 shape 类型，描述多维数组各轴的长度
 */

#pragma once

#include "axis.hpp"

#include <array>
#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <type_traits>

namespace nd {

/**
 * @brief 固定秩形状对象
 *
 * @tparam Rank 维度数，必须在编译期确定且大于 0
 *
 * @details
 * `shape<Rank>` 是整个库最基础的元数据类型之一，用于描述每个轴的长度。
 * 它本身不持有任何数据，只负责保存各轴 extent，并为后续的 stride 推导、
 * 连续性判断、索引映射和结果形状构造提供统一表示。
 */
template <std::size_t Rank>
class shape {
    static_assert(Rank > 0, "Rank must be at least 1");

    std::array<index_t, Rank> dims_{};

public:
    static constexpr std::size_t rank = Rank;

    // ---- 构造 ----

    /// @brief 默认构造为全 0 维度的 shape
    constexpr shape() = default;

    /// @brief 由 `std::array` 显式构造
    /// @param dims 各轴长度数组
    constexpr shape(const std::array<index_t, Rank>& dims) : dims_(dims) {}

    /**
     * @brief 由变参维度构造 shape
     * @param args 各轴长度，参数个数必须与 Rank 相同
     */
    template <typename... Args,
              std::enable_if_t<sizeof...(Args) == Rank &&
                               std::conjunction_v<std::is_convertible<Args, index_t>...>, int> = 0>
    constexpr shape(Args... args) : dims_{static_cast<index_t>(args)...} {}

    /**
     * @brief 由 `initializer_list<int>` 构造
     * @param il 形如 `{4, 8, 16}` 的维度列表
     *
     * @details
     * 该重载主要用于提升易用性，但在 MSVC 下需要注意避免与 `size_t`
     * 场景发生窄化初始化歧义。
     */
    shape(std::initializer_list<int> il) {
        assert(il.size() == Rank);
        auto it = il.begin();
        for (std::size_t i = 0; i < Rank; ++i) {
            assert(*it >= 0 && "Shape dimensions must be non-negative");
            dims_[i] = static_cast<index_t>(*(it++));
        }
    }

    // ---- 元素访问 ----

    /// @brief 读取第 i 个轴的长度
    /// @param i 轴下标
    /// @return 第 i 个轴的 extent
    constexpr index_t  operator[](std::size_t i) const { return dims_[i]; }

    /// @brief 读写第 i 个轴的长度
    /// @param i 轴下标
    /// @return 第 i 个轴长度的可写引用
    constexpr index_t& operator[](std::size_t i)       { return dims_[i]; }

    /// @brief 以 `std::array` 形式访问底层存储
    constexpr const std::array<index_t, Rank>& raw() const { return dims_; }

    /// @brief 以 `std::array` 形式访问底层存储（可写）
    constexpr       std::array<index_t, Rank>& raw()       { return dims_; }

    // ---- 查询 ----

    /**
     * @brief 计算所有轴长度之积
     * @return 元素总数
     *
     * @details
     * 这是静态秩容器中最常用的元数据操作之一，用于构造底层缓冲区大小，
     * 以及在 reshape / flatten / reduce 等流程中校验元素总量一致性。
     */
    constexpr index_t total_size() const {
        index_t s = 1;
        for (std::size_t i = 0; i < Rank; ++i) s *= dims_[i];
        return s;
    }

    // ---- 迭代 ----

    constexpr auto begin() const { return dims_.begin(); }
    constexpr auto end()   const { return dims_.end();   }
    constexpr auto begin()       { return dims_.begin(); }
    constexpr auto end()         { return dims_.end();   }

    // ---- 比较 ----

    /// @brief 比较两个 shape 是否逐轴相等
    constexpr bool operator==(const shape& o) const { return dims_ == o.dims_; }

    /// @brief 比较两个 shape 是否存在任一轴不相等
    constexpr bool operator!=(const shape& o) const { return dims_ != o.dims_; }
};

} // namespace nd

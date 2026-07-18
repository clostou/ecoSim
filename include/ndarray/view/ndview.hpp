/**
 * @file ndview.hpp
 * @brief 非拥有多维视图：基于 shape / stride / 基指针的仿射访问
 */

#pragma once

#include "../core/axis.hpp"
#include "../core/shape.hpp"
#include "../core/stride.hpp"
#include "../core/layout.hpp"
#include "../core/utility.hpp"

#include <algorithm>
#include <type_traits>

namespace nd {

/**
 * @brief 多维数组的非拥有视图
 *
 * ndview 不管理底层内存的生命周期，仅通过 data 指针、shape 和 strides
 * 描述一块数据的访问方式。切片、转置等操作产生的新 ndview 共享同一块
 * 底层存储，实现零拷贝。
 *
 * @tparam T    元素类型（可以是 const T 以表示只读视图）
 * @tparam Rank 维度数（编译期固定）
 */
template <typename T, std::size_t Rank>
class ndview {
    static_assert(Rank > 0, "Rank must be at least 1");

    T*             data_    = nullptr;
    shape<Rank>    shape_{};
    strides<Rank>  strides_{};

public:
    using value_type = T;
    static constexpr std::size_t rank = Rank;

    // ---- 构造 ----

    /// @brief 默认构造为空视图
    ndview() = default;

    /**
     * @brief 由数据指针、shape 和 stride 构造视图
     * @param data 数据首地址
     * @param sh 视图形状
     * @param st 视图步长
     */
    ndview(T* data, const shape<Rank>& sh, const strides<Rank>& st)
        : data_(data), shape_(sh), strides_(st) {}

    /**
     * @brief 由数据指针和 shape 构造连续视图
     * @param data 数据首地址
     * @param sh 视图形状
     *
     * @details
     * 该构造函数默认按 row-major 推导 stride，适合包装连续内存。
     */
    ndview(T* data, const shape<Rank>& sh)
        : data_(data), shape_(sh), strides_(row_major::compute_strides(sh)) {}

    // ---- 标量索引 ----

    /**
     * @brief 访问单个元素（可写）
     * @param indices 每个轴的下标，支持负下标规范化
     * @return 对应元素引用
     */
    template <typename... Indices,
              std::enable_if_t<sizeof...(Indices) == Rank &&
                               std::conjunction_v<std::is_integral<Indices>...>, int> = 0>
    T& operator()(Indices... indices) {
        check_bounds(shape_, indices...);
        return data_[compute_offset_norm(shape_, strides_, indices...)];
    }

    /**
     * @brief 访问单个元素（只读）
     * @param indices 每个轴的下标，支持负下标规范化
     * @return 对应元素常量引用
     */
    template <typename... Indices,
              std::enable_if_t<sizeof...(Indices) == Rank &&
                               std::conjunction_v<std::is_integral<Indices>...>, int> = 0>
    const T& operator()(Indices... indices) const {
        check_bounds(shape_, indices...);
        return data_[compute_offset_norm(shape_, strides_, indices...)];
    }

    // ---- 元数据查询 ----

    /// @brief 返回完整 shape 对象
    const nd::shape<Rank>&   shape()               const { return shape_;   }
    /// @brief 返回指定轴长度
    index_t                  shape(axis_t ax)       const { return shape_[ax]; }
    /// @brief 返回完整 stride 对象
    const nd::strides<Rank>& strides()              const { return strides_; }
    /// @brief 返回指定轴的 stride
    sindex_t                 stride(axis_t ax)      const { return strides_[ax]; }
    /// @brief 返回逻辑元素总数
    index_t                  size()                 const { return shape_.total_size(); }
    /// @brief 返回维度数
    static constexpr std::size_t ndim()                   { return Rank; }

    /// @brief 返回可写数据指针
    T*       data()       { return data_; }
    /// @brief 返回只读数据指针
    const T* data() const { return data_; }

    /**
     * @brief 判断当前视图是否仍为 row-major 连续布局
     * @return 若 shape 与 stride 匹配默认连续布局则返回 true
     */
    bool is_contiguous() const { return nd::is_contiguous(shape_, strides_); }

    // ---- 批量写入 ----

    /**
     * @brief 将整个视图填充为同一个值
     * @param value 填充值
     *
     * @details
     * 连续视图走 `std::fill_n` 快路径；非连续视图则递归遍历所有逻辑坐标，
     * 通过 stride 完成逐元素写入。
     */
    void fill(const T& value) {
        if (is_contiguous()) {
            std::fill_n(data_, size(), value);
        } else {
            fill_recursive<0>(value, 0);
        }
    }

    // ---- 只读 → 可写 转换阻断 ----
    // const T 类型的 ndview 不允许写入（由编译器静态保证）

private:
    /**
     * @brief 非连续视图的递归填充回退实现
     * @tparam Dim 当前递归处理的轴
     * @param value 填充值
     * @param base 当前递归基址偏移
     */
    template <std::size_t Dim>
    void fill_recursive(const T& value, sindex_t base) {
        if constexpr (Dim == Rank - 1) {
            for (index_t i = 0; i < shape_[Dim]; ++i)
                data_[base + static_cast<sindex_t>(i) * strides_[Dim]] = value;
        } else {
            for (index_t i = 0; i < shape_[Dim]; ++i)
                fill_recursive<Dim + 1>(value,
                    base + static_cast<sindex_t>(i) * strides_[Dim]);
        }
    }
};

} // namespace nd

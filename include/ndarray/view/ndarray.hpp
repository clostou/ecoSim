/**
 * @file ndarray.hpp  (view/ndarray.hpp)
 * @brief 拥有数据的多维数组主容器及工厂函数
 */

#pragma once

#include "ndview.hpp"
#include "../storage/buffer.hpp"
#include "../core/traits.hpp"

#include <cmath>
#include <cassert>
#include <type_traits>

namespace nd {

/**
 * @brief 拥有底层存储的多维数组
 *
 * ndarray 持有一块连续分配的内存，并通过 shape / strides 解释为多维数组。
 * 默认使用 row-major 布局与对齐分配器。
 *
 * @tparam T         元素类型（必须是 C++ 基本算术类型）
 * @tparam Rank      维度数（编译期固定）
 * @tparam Layout    布局策略（默认 row_major）
 * @tparam Allocator 分配器类型（默认 aligned_allocator<T>）
 */
template <typename T, std::size_t Rank,
          typename Layout    = row_major,
          typename Allocator = aligned_allocator<T>>
class ndarray {
    static_assert(Rank > 0,          "Rank must be at least 1");
    static_assert(is_scalar_v<T>,    "Element type must be a scalar arithmetic type");

    buffer<T, Allocator>  buf_;
    nd::shape<Rank>       shape_{};
    nd::strides<Rank>     strides_{};

public:
    using value_type     = T;
    using layout_type    = Layout;
    using allocator_type = Allocator;
    static constexpr std::size_t rank = Rank;

    // ========== 构造 ==========

    /// @brief 默认构造为空数组对象
    ndarray() = default;

    /**
     * @brief 由 shape 构造连续数组
     * @param sh 数组形状
     *
     * @details
     * 底层缓冲区大小等于 `sh.total_size()`，stride 由布局策略自动推导。
     */
    explicit ndarray(const nd::shape<Rank>& sh)
        : buf_(sh.total_size())
        , shape_(sh)
        , strides_(Layout::compute_strides(sh)) {}

    /// @brief 由 shape 构造并以给定值填充
    ndarray(const nd::shape<Rank>& sh, const T& value)
        : buf_(sh.total_size(), value)
        , shape_(sh)
        , strides_(Layout::compute_strides(sh)) {}

    /// @brief 由变参维度构造连续数组
    template <typename... Dims,
              std::enable_if_t<sizeof...(Dims) == Rank &&
                               std::conjunction_v<std::is_integral<Dims>...>, int> = 0>
    explicit ndarray(Dims... dims)
        : ndarray(nd::shape<Rank>(dims...)) {}

    /// @brief 拷贝构造，底层缓冲区执行深拷贝
    ndarray(const ndarray&)            = default;
    /// @brief 拷贝赋值，底层缓冲区执行深拷贝
    ndarray& operator=(const ndarray&) = default;
    /// @brief 移动构造，转移底层存储所有权
    ndarray(ndarray&&) noexcept        = default;
    /// @brief 移动赋值，转移底层存储所有权
    ndarray& operator=(ndarray&&) noexcept = default;

    // ========== 标量索引 ==========

    /// @brief 访问单个元素（可写），支持负下标规范化
    template <typename... Indices,
              std::enable_if_t<sizeof...(Indices) == Rank &&
                               std::conjunction_v<std::is_integral<Indices>...>, int> = 0>
    T& operator()(Indices... indices) {
        check_bounds(shape_, indices...);
        return buf_[static_cast<index_t>(compute_offset_norm(shape_, strides_, indices...))];
    }

    /// @brief 访问单个元素（只读），支持负下标规范化
    template <typename... Indices,
              std::enable_if_t<sizeof...(Indices) == Rank &&
                               std::conjunction_v<std::is_integral<Indices>...>, int> = 0>
    const T& operator()(Indices... indices) const {
        check_bounds(shape_, indices...);
        return buf_[static_cast<index_t>(compute_offset_norm(shape_, strides_, indices...))];
    }

    // ========== 元数据 ==========

    /// @brief 返回完整 shape 对象
    const nd::shape<Rank>&   shape()            const { return shape_;   }
    /// @brief 返回指定轴长度
    index_t                  shape(axis_t ax)   const { return shape_[ax]; }
    /// @brief 返回完整 stride 对象
    const nd::strides<Rank>& strides()          const { return strides_; }
    /// @brief 返回指定轴的 stride
    sindex_t                 stride(axis_t ax)  const { return strides_[ax]; }
    /// @brief 返回逻辑元素总数
    index_t                  size()             const { return shape_.total_size(); }
    /// @brief 返回维度数
    static constexpr std::size_t ndim()               { return Rank; }

    /// @brief 返回可写数据指针
    T*       data()       { return buf_.data(); }
    /// @brief 返回只读数据指针
    const T* data() const { return buf_.data(); }

    /// ndarray 始终连续
    bool is_contiguous() const { return true; }

    // ========== 复制与填充 ==========

    /// @brief 深拷贝整个数组对象
    ndarray clone() const { return ndarray(*this); }
    /// @brief 以统一值填充整个数组
    void    fill(const T& value) { buf_.fill(value); }
    /// @brief 将整个数组置零
    void    zero()               { buf_.zero(); }

    // ========== 视图转换 ==========

    /// @brief 隐式转换为可写 `ndview`
    operator ndview<T, Rank>() {
        return ndview<T, Rank>(buf_.data(), shape_, strides_);
    }
    /// @brief 隐式转换为只读 `ndview`
    operator ndview<const T, Rank>() const {
        return ndview<const T, Rank>(buf_.data(), shape_, strides_);
    }

    /// @brief 获取可写视图
    ndview<T, Rank>       view()       { return *this; }
    /// @brief 获取只读视图
    ndview<const T, Rank> view() const { return *this; }
    /// @brief 获取显式常量视图
    ndview<const T, Rank> cview()const { return *this; }
};

// ===============================================================
//  工厂函数
// ===============================================================

/// @brief 创建未初始化的连续数组
template <typename T, std::size_t Rank>
ndarray<T, Rank> empty(const shape<Rank>& sh) {
    return ndarray<T, Rank>(sh);
}

/// @brief 由变参数维度创建未初始化数组
template <typename T, typename... Dims,
          std::enable_if_t<std::conjunction_v<std::is_convertible<Dims, index_t>...> &&
                           sizeof...(Dims) >= 1, int> = 0>
auto empty(Dims... dims) {
    return ndarray<T, sizeof...(Dims)>(shape<sizeof...(Dims)>(dims...));
}

/// @brief 创建全 0 数组
template <typename T, std::size_t Rank>
ndarray<T, Rank> zeros(const shape<Rank>& sh) {
    ndarray<T, Rank> a(sh);
    a.zero();
    return a;
}

/// @brief 由变参数维度创建全 0 数组
template <typename T, typename... Dims,
          std::enable_if_t<std::conjunction_v<std::is_convertible<Dims, index_t>...> &&
                           sizeof...(Dims) >= 1, int> = 0>
auto zeros(Dims... dims) {
    return zeros<T>(shape<sizeof...(Dims)>(dims...));
}

/// @brief 创建全 1 数组
template <typename T, std::size_t Rank>
ndarray<T, Rank> ones(const shape<Rank>& sh) {
    return ndarray<T, Rank>(sh, T{1});
}

/// @brief 由变参数维度创建全 1 数组
template <typename T, typename... Dims,
          std::enable_if_t<std::conjunction_v<std::is_convertible<Dims, index_t>...> &&
                           sizeof...(Dims) >= 1, int> = 0>
auto ones(Dims... dims) {
    return ones<T>(shape<sizeof...(Dims)>(dims...));
}

/// @brief 创建以指定值填充的数组
template <typename T, std::size_t Rank>
ndarray<T, Rank> full(const shape<Rank>& sh, const T& value) {
    return ndarray<T, Rank>(sh, value);
}

/// @brief 由变参数维度创建指定值数组
template <typename T, typename... Dims,
          std::enable_if_t<std::conjunction_v<std::is_convertible<Dims, index_t>...> &&
                           sizeof...(Dims) >= 1, int> = 0>
auto full(const T& value, Dims... dims) {
    return full<T>(shape<sizeof...(Dims)>(dims...), value);
}

/**
 * @brief 创建一维等差序列
 * @param start 起始值（包含）
 * @param stop 终止值（不包含）
 * @param step 步长，不能为 0
 */
template <typename T>
ndarray<T, 1> arange(T start, T stop, T step = T{1}) {
    assert(step != T{0} && "Step must not be zero");
    index_t count = 0;
    if ((step > T{0} && stop > start) || (step < T{0} && stop < start)) {
        count = static_cast<index_t>(
            std::ceil(static_cast<double>(stop - start) /
                      static_cast<double>(step)));
    }
    ndarray<T, 1> result{shape<1>(count)};
    for (index_t i = 0; i < count; ++i)
        result(i) = start + static_cast<T>(i) * step;
    return result;
}

/**
 * @brief 创建一维线性等分序列
 * @param start 起始值（包含）
 * @param stop 终止值（包含）
 * @param count 样本个数
 */
template <typename T>
ndarray<T, 1> linspace(T start, T stop, index_t count) {
    ndarray<T, 1> result{shape<1>(count)};
    if (count == 0) return result;
    if (count == 1) { result(0) = start; return result; }
    const T step = (stop - start) / static_cast<T>(count - 1);
    for (index_t i = 0; i < count; ++i)
        result(i) = start + static_cast<T>(i) * step;
    return result;
}

/// @brief 从外部连续缓冲区复制构造新数组
template <typename T, std::size_t Rank>
ndarray<T, Rank> from_buffer(const T* data, const shape<Rank>& sh) {
    ndarray<T, Rank> result(sh);
    std::copy_n(data, sh.total_size(), result.data());
    return result;
}

/// @brief 以另一个数组的形状创建全 0 数组
template <typename T, std::size_t R, typename L, typename A>
ndarray<T, R> zeros_like(const ndarray<T, R, L, A>& other) {
    return zeros<T>(other.shape());
}

/// @brief 以另一个数组的形状创建全 1 数组
template <typename T, std::size_t R, typename L, typename A>
ndarray<T, R> ones_like(const ndarray<T, R, L, A>& other) {
    return ones<T>(other.shape());
}

/// @brief 以另一个数组的形状创建指定值数组
template <typename T, std::size_t R, typename L, typename A>
ndarray<T, R> full_like(const ndarray<T, R, L, A>& other, const T& value) {
    return full<T>(other.shape(), value);
}

/// @brief 常用别名：一维向量
template <typename T> using vector  = ndarray<T, 1>;
/// @brief 常用别名：二维矩阵
template <typename T> using matrix  = ndarray<T, 2>;
/// @brief 常用别名：三维张量
template <typename T> using tensor3 = ndarray<T, 3>;

} // namespace nd

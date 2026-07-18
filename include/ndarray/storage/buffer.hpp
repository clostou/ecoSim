/**
 * @file buffer.hpp
 * @brief 拥有连续内存的 RAII 缓冲区
 */

#pragma once

#include "allocator.hpp"

#include <algorithm>
#include <cstring>

namespace nd {

/**
 * @brief 一维连续存储缓冲区，支持 RAII 与深拷贝
 * @tparam T         元素类型
 * @tparam Allocator 分配器类型
 */
template <typename T, typename Allocator = aligned_allocator<T>>
class buffer {
    T*          data_  = nullptr;
    std::size_t size_  = 0;
    Allocator   alloc_;

public:
    // ---- 构造 / 析构 ----

    /// @brief 默认构造为空缓冲区
    buffer() = default;

    /**
     * @brief 分配指定大小的连续缓冲区
     * @param size 元素个数
     * @param alloc 分配器实例
     *
     * @details
     * 为了保持 header-only 和低开销，这里不逐元素构造复杂对象；当前库的元素类型
     * 被限制为标量，因此可以安全地把该缓冲区视为原始连续内存。
     */
    explicit buffer(std::size_t size, const Allocator& alloc = Allocator())
        : size_(size), alloc_(alloc)
    {
        if (size_ > 0)
            data_ = alloc_.allocate(size_);
    }

    /// @brief 分配并以同一个值填充整个缓冲区
    buffer(std::size_t size, const T& value, const Allocator& alloc = Allocator())
        : buffer(size, alloc)
    {
        std::fill_n(data_, size_, value);
    }

    /// @brief 析构时释放持有的连续内存
    ~buffer() {
        if (data_)
            alloc_.deallocate(data_, size_);
    }

    // ---- 移动 ----

    /// @brief 移动构造，转移底层指针所有权
    buffer(buffer&& o) noexcept
        : data_(o.data_), size_(o.size_), alloc_(std::move(o.alloc_))
    {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    /// @brief 移动赋值，释放旧资源后接管新资源
    buffer& operator=(buffer&& o) noexcept {
        if (this != &o) {
            if (data_) alloc_.deallocate(data_, size_);
            data_  = o.data_;
            size_  = o.size_;
            alloc_ = std::move(o.alloc_);
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    // ---- 拷贝（深拷贝）----

    /// @brief 深拷贝构造
    buffer(const buffer& o)
        : size_(o.size_), alloc_(o.alloc_)
    {
        if (size_ > 0) {
            data_ = alloc_.allocate(size_);
            std::copy_n(o.data_, size_, data_);
        }
    }

    /// @brief 深拷贝赋值，采用 copy-swap 保持异常安全
    buffer& operator=(const buffer& o) {
        if (this != &o) {
            buffer tmp(o);
            swap(tmp);
        }
        return *this;
    }

    // ---- 访问 ----

    /// @brief 返回可写数据指针
    T*          data()       noexcept { return data_; }

    /// @brief 返回只读数据指针
    const T*    data() const noexcept { return data_; }

    /// @brief 返回元素个数
    std::size_t size() const noexcept { return size_; }

    /// @brief 判断缓冲区是否为空
    bool        empty()const noexcept { return size_ == 0; }

    /// @brief 按一维线性下标访问元素
    T&       operator[](std::size_t i)       { return data_[i]; }

    /// @brief 按一维线性下标只读访问元素
    const T& operator[](std::size_t i) const { return data_[i]; }

    // ---- 批量操作 ----

    /// @brief 以同一个值填充整个缓冲区
    void fill(const T& value)   { std::fill_n(data_, size_, value); }

    /// @brief 将缓冲区按字节清零
    void zero()                 { if (data_) std::memset(data_, 0, size_ * sizeof(T)); }

    /// @brief 交换两个缓冲区的所有权与分配器状态
    void swap(buffer& o) noexcept {
        std::swap(data_,  o.data_);
        std::swap(size_,  o.size_);
        std::swap(alloc_, o.alloc_);
    }
};

} // namespace nd

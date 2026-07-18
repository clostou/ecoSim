/**
 * @file allocator.hpp
 * @brief 对齐内存分配器，默认 64 字节对齐（缓存行友好 / AVX-512 兼容）
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace nd {

/// 默认对齐字节数
inline constexpr std::size_t default_alignment = 64;

/**
 * @brief 对齐分配器（满足 Allocator 具名要求）
 * @tparam T         元素类型
 * @tparam Alignment 对齐字节数
 */
template <typename T, std::size_t Alignment = default_alignment>
class aligned_allocator {
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

        /// @brief 默认构造
    aligned_allocator() noexcept = default;

        /// @brief 允许不同 value_type 间共享同一对齐策略
    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

        /**
         * @brief 分配 n 个按 Alignment 对齐的元素空间
         * @param n 元素个数
         * @return 对齐后的内存首地址
         */
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

        /**
         * @brief 释放由当前分配器分配的内存
         * @param p 内存首地址
         */
    void deallocate(T* p, std::size_t) noexcept {
        if (!p) return;
#ifdef _MSC_VER
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

        /// @brief `Allocator` 具名要求中的重绑定定义
    template <typename U>
    struct rebind { using other = aligned_allocator<U, Alignment>; };

        /// @brief 相同对齐策略的分配器视为总是相等
    bool operator==(const aligned_allocator&) const noexcept { return true; }

        /// @brief 与 `operator==` 对偶的比较接口
    bool operator!=(const aligned_allocator&) const noexcept { return false; }
};

} // namespace nd

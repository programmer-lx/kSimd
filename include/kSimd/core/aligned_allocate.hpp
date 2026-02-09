#pragma once

#include <cstdlib>

#include <type_traits>
#include <new>

#include "impl/common_macros.hpp"
#include "impl/platform.hpp"


namespace ksimd
{
    KSIMD_HEADER_GLOBAL void* aligned_allocate(size_t bytes, size_t alignment) noexcept
    {
    #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        return _aligned_malloc(bytes, alignment);
        #else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, bytes) != 0)
        {
            return nullptr;
        }
        return ptr;
        #endif
    }

    KSIMD_HEADER_GLOBAL void aligned_free(void* mem) noexcept
    {
        #if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
        _aligned_free(mem);
        #else
        free(mem);
        #endif
    }


    // 以最大对齐字节进行分配
    template<typename T>
    struct AlignedAllocator
    {
        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_function_v<T>);
        static_assert(!std::is_reference_v<T>);

        using value_type      = T;
        using size_type       = size_t;
        using difference_type = ptrdiff_t;

        AlignedAllocator() noexcept = default;

        AlignedAllocator(const AlignedAllocator&) noexcept = default;

        template <class Other>
        AlignedAllocator(const AlignedAllocator<Other>&) noexcept {}

        ~AlignedAllocator() = default;

        AlignedAllocator& operator=(const AlignedAllocator&) = default;

        [[nodiscard]] T* allocate(const size_t count)
        {
            const size_t bytes = count * sizeof(T);
            void* ptr = aligned_allocate(bytes, alignment::Max);

            if (!ptr)
            {
                throw std::bad_alloc();
            }

            return static_cast<T*>(ptr);
        }

        void deallocate(T* const mem, const size_t)
        {
            aligned_free(mem);
        }
    };
}

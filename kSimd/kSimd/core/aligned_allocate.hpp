#pragma once

#include <cstdlib> // aligned malloc

#include <type_traits>
#include <new>

#include "impl/base.hpp"
#include "impl/types.hpp"

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


    template<typename T, size_t Alignment = alignment::Required>
    struct AlignedAllocator
    {
        static_assert(!std::is_const_v<T>);
        static_assert(!std::is_function_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of two");
        static_assert(Alignment >= alignment::Vec128, "Alignment must be >= alignof(Vec128)");

        using value_type        = T;
        using size_type         = size_t;
        using difference_type   = ptrdiff_t;

        AlignedAllocator() noexcept = default;

        AlignedAllocator(const AlignedAllocator&) noexcept = default;

        template <class Other>
        AlignedAllocator(const AlignedAllocator<Other, Alignment>&) noexcept {}

        ~AlignedAllocator() = default;

        template<class U>
        struct rebind
        {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator& operator=(const AlignedAllocator&) = default;

        [[nodiscard]] T* allocate(const size_t count)
        {
            if (count > std::numeric_limits<size_type>::max() / sizeof(T))
                throw std::bad_alloc();

            const size_t bytes = count * sizeof(T);
            void* ptr = aligned_allocate(bytes, Alignment);

            if (!ptr)
            {
                throw std::bad_alloc();
            }

            return static_cast<T*>(ptr);
        }

        void deallocate(T* const mem, const size_t) noexcept
        {
            aligned_free(mem);
        }
    };
}

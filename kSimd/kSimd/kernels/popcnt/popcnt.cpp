#define KSIMD_KERNEL_POPCNT_EXPORT
#include "kSimd/kernels/popcnt/popcnt.h"

#include <immintrin.h>
#include <cstring> // memcpy

#include "kSimd/core/impl/dispatch.hpp"

namespace
{
    ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer_soft(const void* buffer, ks_bytesize_t byte_size) noexcept
    {
        ks_pop_bitcount_t cnt  = 0;
        ks_pop_bitcount_t cnt1 = 0;
        ks_pop_bitcount_t cnt2 = 0;
        ks_pop_bitcount_t cnt3 = 0;
        ks_pop_bitcount_t cnt4 = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;
        const size_t size = static_cast<size_t>(byte_size);

        #if KSIMD_ARCH_X86_64
        // for each u64 x 4
        for (; i + 32 <= size; i += 32)
        {
            cnt1 += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i     ));
            cnt2 += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i +  8));
            cnt3 += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i + 16));
            cnt4 += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i + 24));
        }

        // for each u64
        for (; i + 8 <= size; i += 8)
        {
            cnt += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i));
        }
        #elif KSIMD_ARCH_X86_32
        // for each u32 x 4
        for (; i + 16 <= size; i += 16)
        {
            cnt1 += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i     ));
            cnt2 += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i +  4));
            cnt3 += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i +  8));
            cnt4 += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i + 12));
        }

        // for each u32
        for (; i + 4 <= size; i += 4)
        {
            cnt += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i));
        }
        #else
        #error unknown x86 arch
        #endif

        // for each rest u8
        for (; i < size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt + cnt1 + cnt2 + cnt3 + cnt4;
    }

    KSIMD_DYN_FUNC_ATTR_POPCNT ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV
    ks_popcnt_buffer_x86_popcnt(const void* buffer, ks_bytesize_t byte_size) noexcept
    {
        ks_pop_bitcount_t cnt  = 0;
        ks_pop_bitcount_t cnt1 = 0;
        ks_pop_bitcount_t cnt2 = 0;
        ks_pop_bitcount_t cnt3 = 0;
        ks_pop_bitcount_t cnt4 = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;
        const size_t size = static_cast<size_t>(byte_size);

        #if KSIMD_ARCH_X86_64
        // for each u64 x 4
        for (; i + 32 <= size; i += 32)
        {
            uint64_t v1;
            uint64_t v2;
            uint64_t v3;
            uint64_t v4;

            std::memcpy(&v1, data + i     , 8);
            std::memcpy(&v2, data + i +  8, 8);
            std::memcpy(&v3, data + i + 16, 8);
            std::memcpy(&v4, data + i + 24, 8);

            cnt1 += _mm_popcnt_u64(v1);
            cnt2 += _mm_popcnt_u64(v2);
            cnt3 += _mm_popcnt_u64(v3);
            cnt4 += _mm_popcnt_u64(v4);
        }

        // for each u64
        for (; i + 8 <= size; i += 8)
        {
            uint64_t u64;
            std::memcpy(&u64, data + i, 8);
            cnt += _mm_popcnt_u64(u64);
        }
        #elif KSIMD_ARCH_X86_32
        // for each u32 x 4
        for (; i + 16 <= size; i += 16)
        {
            uint32_t v1;
            uint32_t v2;
            uint32_t v3;
            uint32_t v4;

            std::memcpy(&v1, data + i     , 4);
            std::memcpy(&v2, data + i +  4, 4);
            std::memcpy(&v3, data + i +  8, 4);
            std::memcpy(&v4, data + i + 12, 4);

            cnt1 += _mm_popcnt_u32(v1);
            cnt2 += _mm_popcnt_u32(v2);
            cnt3 += _mm_popcnt_u32(v3);
            cnt4 += _mm_popcnt_u32(v4);
        }

        // for each u32
        for (; i + 4 <= size; i += 4)
        {
            uint32_t u64;
            std::memcpy(&u64, data + i, 4);
            cnt += _mm_popcnt_u32(u64);
        }
        #else
        #error unknown x86 arch
        #endif

        // for each rest u8
        for (; i < size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt + cnt1 + cnt2 + cnt3 + cnt4;
    }

    auto ks_popcnt_fn()
    {
        static auto fn = []()
        {
            const ksimd::CpuSupportInfo& support = ksimd::get_cpu_support_info();

            if (support.POPCNT)
            {
                return ks_popcnt_buffer_x86_popcnt;
            }

            // fallback to soft
            return ks_popcnt_buffer_soft;
        }();

        return fn;
    }
}

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_fn()(buffer, byte_size);
}


/* for testing */
#ifdef KSIMD_IS_TESTING

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_buffer_soft(buffer, byte_size);
}

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86_popcnt(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_buffer_x86_popcnt(buffer, byte_size);
}

#endif

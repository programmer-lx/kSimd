#define KSIMD_KERNEL_POPCNT_EXPORT
#include "kSimd/kernels/popcnt/popcnt.h"

#include <immintrin.h>
#include <cstring> // memcpy

#include "kSimd/core/impl/dispatch.hpp"

namespace
{
    size_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer_soft(const void* buffer, size_t byte_size) noexcept
    {
        size_t cnt = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;

        // for each u64
        for (; i + 8 <= byte_size; i += 8)
        {
            cnt += ks_popcnt64_soft(*reinterpret_cast<const uint64_t*>(data + i));
        }

        // for each rest u8
        for (; i < byte_size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt;
    }

    KSIMD_DYN_FUNC_ATTR_POPCNT size_t KSIMD_KERNEL_CALL_CONV
    ks_popcnt_buffer_x86_popcnt(const void* buffer, size_t byte_size) noexcept
    {
        size_t cnt  = 0;
        size_t cnt1 = 0;
        size_t cnt2 = 0;
        size_t cnt3 = 0;
        size_t cnt4 = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;

        // for each u64 x 4
        for (; i + 32 <= byte_size; i += 32)
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
        for (; i + 8 <= byte_size; i += 8)
        {
            uint64_t u64;
            std::memcpy(&u64, data + i, 8);
            cnt += _mm_popcnt_u64(u64);
        }

        // for each rest u8
        for (; i < byte_size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt + cnt1 + cnt2 + cnt3 + cnt4;
    }

    KSIMD_DYN_FUNC_ATTR_AVX2 size_t KSIMD_KERNEL_CALL_CONV
    ks_popcnt_buffer_x86_avx2(const void* buffer, size_t byte_size) noexcept
    {
        return 0;
    }

    auto ks_popcnt_fn = []()
    {
        const ksimd::CpuSupportInfo& support = ksimd::get_cpu_support_info();

        if (support.POPCNT)
        {
            return ks_popcnt_buffer_x86_popcnt;
        }

        // fallback to soft
        return ks_popcnt_buffer_soft;
    }();
}

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer(const void* buffer, size_t byte_size)
{
    return ks_popcnt_fn(buffer, byte_size);
}


/* for testing */
#ifdef KSIMD_IS_TESTING

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, size_t byte_size)
{
    return ks_popcnt_buffer_soft(buffer, byte_size);
}

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86_popcnt(const void* buffer, size_t byte_size)
{
    return ks_popcnt_buffer_x86_popcnt(buffer, byte_size);
}

#endif

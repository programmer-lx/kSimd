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

        // for u32 (only one)
        if (i + 4 <= byte_size)
        {
            cnt += ks_popcnt32_soft(*reinterpret_cast<const uint32_t*>(data + i));

            i += 4;
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
        size_t cnt = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;

        // for each u64
        for (; i + 8 <= byte_size; i += 8)
        {
            uint64_t u64;
            std::memcpy(&u64, data + i, 8);
            cnt += _mm_popcnt_u64(u64);
        }

        // for rest u32 (only one)
        if (i + 4 <= byte_size)
        {
            uint32_t u32;
            std::memcpy(&u32, data + i, 4);
            cnt += _mm_popcnt_u32(u32);

            i += 4;
        }

        // for each rest u8
        for (; i < byte_size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt;
    }
}

KSIMD_KERNEL_POPCNT_API ks_pfn_popcnt_buffer_t ks_popcnt_buffer = []() -> ks_pfn_popcnt_buffer_t
{
    const ksimd::CpuSupportInfo& support = ksimd::get_cpu_support_info();

    if (support.POPCNT)
    {
        return ks_popcnt_buffer_x86_popcnt;
    }

    // fallback to soft
    return ks_popcnt_buffer_soft;
}();


/* for testing */
#ifdef KSIMD_KERNEL_IS_TESTING

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, size_t byte_size)
{
    return ks_popcnt_buffer_soft(buffer, byte_size);
}

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86_popcnt(const void* buffer, size_t byte_size)
{
    return ks_popcnt_buffer_x86_popcnt(buffer, byte_size);
}

#endif

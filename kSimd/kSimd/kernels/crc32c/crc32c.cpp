#define KSIMD_KERNEL_CRC32C_EXPORT
#include "kSimd/kernels/crc32c/crc32c.h"

#include <cstring> // memcpy
#include <nmmintrin.h> // SSE4.2 CRC32 intrinsic
#include <array> // CRC32C table

#include "kSimd/core/impl/dispatch.hpp"

namespace
{
    bool KSIMD_KERNEL_CALL_CONV support_crc32_intrinsic() noexcept
    {
        uint32_t abcd[4]{};
        ksimd::detail::cpuid(0, 0, abcd);
        const uint32_t max_leaf = abcd[0];
        if (max_leaf >= 1)
        {
            ksimd::detail::cpuid(1, 0, abcd);
            // SSE4.2: EAX 1, ECX 20
            const uint32_t ecx = abcd[2];
            return (ecx & (1 << 20)) != 0;
        }
        return false;
    }
    
    consteval std::array<uint32_t, 256> make_crc32c_table()
    {
        constexpr uint32_t POLY = 0x82F63B78; // 反转后的多项式
        std::array<uint32_t, 256> table{};
        for (uint32_t i = 0; i < 256; i++)
        {
            uint32_t c = i;
            for (int j = 0; j < 8; j++)
            {
                if (c & 1)
                    c = (c >> 1) ^ POLY;
                else
                    c >>= 1;
            }
            table[i] = c;
        }
        return table;
    }

    auto crc32c_table = make_crc32c_table();

    uint32_t KSIMD_KERNEL_CALL_CONV ks_update_crc32c_soft(
        uint32_t origin,
        const void* data,
        size_t size
    ) noexcept
    {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);

        for (size_t i = 0; i < size; i++)
        {
            uint8_t index = (origin ^ bytes[i]) & 0xff;
            origin = (origin >> 8) ^ crc32c_table[index];
        }

        return origin;
    }

    KSIMD_DYN_FUNC_ATTR_SSE42 uint32_t KSIMD_KERNEL_CALL_CONV ks_update_crc32c_sse42(
        uint32_t origin,
        const void* data,
        size_t size
    ) noexcept
    {
        uint32_t crc = origin;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);

        size_t i = 0;

        #if KSIMD_ARCH_X86_64
        for (; i + 8 <= size; i += 8)
        {
            uint64_t v;
            std::memcpy(&v, bytes + i, sizeof(uint64_t)); // 避免未对齐 UB
            crc = static_cast<uint32_t>(_mm_crc32_u64(crc, v));
        }
        #endif

        for (; i + 4 <= size; i += 4)
        {
            uint32_t v;
            std::memcpy(&v, bytes + i, sizeof(uint32_t));
            crc = _mm_crc32_u32(crc, v);
        }

        for (; i < size; ++i)
        {
            crc = _mm_crc32_u8(crc, bytes[i]);
        }

        return crc;
    }

    auto ks_update_crc32c_fn = []()
    {
        if (support_crc32_intrinsic())
        {
            return ks_update_crc32c_sse42;
        }

        // fallback
        return ks_update_crc32c_soft;
    }();
}

KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_update_crc32c(uint32_t origin, const void* data, size_t size)
{
    return ks_update_crc32c_fn(origin, data, size);
}



// for testing
#ifdef KSIMD_IS_TESTING
KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_test_update_crc32c_soft(
    uint32_t origin,
    const void* data,
    size_t size
)
{
    return ks_update_crc32c_soft(origin, data, size);
}

KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_test_update_crc32c_sse42(
    uint32_t origin,
    const void* data,
    size_t size
)
{
    return ks_update_crc32c_sse42(origin, data, size);
}
#endif
#define KSIMD_KERNEL_POPCNT_EXPORT
#include "kSimd/kernels/popcnt/popcnt.h"

#if KSIMD_ARCH_X86_ANY
    #include <immintrin.h>
#endif

#if KSIMD_ARCH_ARM_ANY
    #include <arm_neon.h>
#endif

#include <algorithm> // std::min
#include <cstring> // memcpy

#include "kSimd/core/impl/dispatch.hpp"

namespace
{
    ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer_soft(const void* buffer, size_t size) noexcept
    {
        ks_pop_bitcount_t cnt  = 0;
        ks_pop_bitcount_t cnt1 = 0;
        ks_pop_bitcount_t cnt2 = 0;
        ks_pop_bitcount_t cnt3 = 0;
        ks_pop_bitcount_t cnt4 = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;

        #if KSIMD_ARCH_X86_64 || KSIMD_ARCH_ARM_64
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
        #else
        #error unknown arch
        #endif

        // for each rest u8
        for (; i < size; ++i)
        {
            cnt += ks_popcnt8_soft(data[i]);
        }

        return cnt + cnt1 + cnt2 + cnt3 + cnt4;
    }

    #if KSIMD_ARCH_X86_ANY
    KSIMD_DYN_FUNC_ATTR_POPCNT ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV
    ks_popcnt_buffer_x86(const void* buffer, size_t size) noexcept
    {
        ks_pop_bitcount_t cnt  = 0;
        ks_pop_bitcount_t cnt1 = 0;
        ks_pop_bitcount_t cnt2 = 0;
        ks_pop_bitcount_t cnt3 = 0;
        ks_pop_bitcount_t cnt4 = 0;

        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);

        size_t i = 0;

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
    #endif

    #if KSIMD_ARCH_ARM_ANY
    KSIMD_DYN_FUNC_ATTR_NEON ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV
    ks_popcnt_buffer_arm_neon(const void* buffer, size_t size) noexcept
    {
        ks_pop_bitcount_t total_cnt = 0;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(buffer);
        size_t i = 0;

        // for each u8 x 16 x 4 (vec128 x 4)
        while (i + 64 <= size)
        {
            uint16x8_t acc_a = vdupq_n_u16(0);
            uint16x8_t acc_b = vdupq_n_u16(0);
            
            // u8的上限是255，如果直接使用u8进行累加的话，最多累加31次，就要立即进行水平求和了
            // u16的上限是65535，直接进行u8累加，可以累加 65536/8=8192 次，但是现在使用一个u16存储两个u8的累加结果
            // 所以可以累加 8192/2-1 = 4095 次，减少了水平求和的次数
            size_t inner_limit = std::min(size - 64, i + 4095 * 64);
            for (; i <= inner_limit; i += 64)
            {
                uint8x16_t c0 = vcntq_u8(vld1q_u8(data + i));
                uint8x16_t c1 = vcntq_u8(vld1q_u8(data + i + 16));
                uint8x16_t c2 = vcntq_u8(vld1q_u8(data + i + 32));
                uint8x16_t c3 = vcntq_u8(vld1q_u8(data + i + 48));

                // 垂直累加：将 8-bit 拓宽到 16-bit
                acc_a = vpadalq_u8(acc_a, c0);
                acc_b = vpadalq_u8(acc_b, c2);
                acc_a = vpadalq_u8(acc_a, c1);
                acc_b = vpadalq_u8(acc_b, c3);
            }
            
            // 使用针对 u16 的水平累加指令
            total_cnt += vaddlvq_u16(acc_a);
            total_cnt += vaddlvq_u16(acc_b);
        }

        // for each u8 x 16 (vec128 x 1)
        // 最多有3组vec128，3 x 8 = 24，所以一定不会溢出，所以最后再累加求和即可
        if (i + 16 <= size)
        {
            uint8x16_t acc = vdupq_n_u8(0);
            for (; i + 16 <= size; i += 16)
            {
                acc = vaddq_u8(acc, vcntq_u8(vld1q_u8(data + i)));
            }
            total_cnt += vaddlvq_u8(acc);
        }

        // for each rest u8
        for (; i < size; ++i)
        {
            total_cnt += ks_popcnt8_soft(data[i]);
        }

        return total_cnt;
    }
    #endif

    auto ks_popcnt_fn()
    {
        static auto fn = []()
        {
            const ksimd::CpuSupportInfo& support = ksimd::get_cpu_support_info();

            #if KSIMD_ARCH_X86_ANY
            if (support.popcnt)
            {
                return ks_popcnt_buffer_x86;
            }
            #endif

            #if KSIMD_ARCH_ARM_ANY
            if (support.neon)
            {
                return ks_popcnt_buffer_arm_neon;
            }
            #endif

            // fallback to soft
            return ks_popcnt_buffer_soft;
        }();

        return fn;
    }
}

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_fn()(buffer, static_cast<size_t>(byte_size));
}


/* for testing */
#ifdef KSIMD_IS_TESTING

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_buffer_soft(buffer, static_cast<size_t>(byte_size));
}

#if KSIMD_ARCH_X86_ANY
KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_buffer_x86(buffer, static_cast<size_t>(byte_size));
}
#endif

#if KSIMD_ARCH_ARM_ANY
KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_arm_neon(const void* buffer, ks_bytesize_t byte_size)
{
    return ks_popcnt_buffer_arm_neon(buffer, static_cast<size_t>(byte_size));
}
#endif

#endif

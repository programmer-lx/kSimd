#pragma once

/*
#define KSIMD_KERNEL_POPCNT_AS_DLL to use this kernel as DLL.

To calculate count of 1bit in a buffer:
ks_bitcount_t count = ks_popcnt_buffer(&buffer, byte size of buffer);

If just calculate the count of 1bit of only one integer, use software simulation:
ks_bitcount_t count = ks_popcnt64_soft(uint64_t x)
ks_bitcount_t count = ks_popcnt32_soft(uint32_t x)
ks_bitcount_t count = ks_popcnt16_soft(uint16_t x)
ks_bitcount_t count = ks_popcnt8_soft(uint8_t x)
*/

#ifdef __cplusplus
    #include <cstdint>
    #include <cstddef>
#else
    #include <stdint.h>
    #include <stddef.h>
#endif

#include "kSimd/kernels/kernel_api.h"

#ifdef KSIMD_KERNEL_POPCNT_AS_DLL
    /* shared lib */
    #ifdef KSIMD_KERNEL_POPCNT_EXPORT /* cpp file should define this macro. */
        #define KSIMD_KERNEL_POPCNT_API KSIMD_KERNEL_DLL_EXPORT
    #else
        #define KSIMD_KERNEL_POPCNT_API KSIMD_KERNEL_DLL_IMPORT
    #endif
#else
    /* static lib */
    #define KSIMD_KERNEL_POPCNT_API
#endif

KSIMD_KERNEL_BEGIN_EXTERN_C

/* 32位操作系统上，如果一个buffer超过了512MB，size_t就会溢出，所以强制使用uint64类型 */
typedef uint64_t ks_pop_bitcount_t;

static KSIMD_KERNEL_FORCE_INLINE ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt64_soft(uint64_t x)
{
    x -= (x >> 1) & UINT64_C(0x5555555555555555);
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (ks_pop_bitcount_t)((x * UINT64_C(0x0101010101010101)) >> 56);
}

static KSIMD_KERNEL_FORCE_INLINE ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt32_soft(uint32_t x)
{
    x -= (x >> 1) & UINT32_C(0x55555555);
    x = (x & UINT32_C(0x33333333)) + ((x >> 2) & UINT32_C(0x33333333));
    x = (x + (x >> 4)) & UINT32_C(0x0f0f0f0f);
    return (ks_pop_bitcount_t)((x * UINT32_C(0x01010101)) >> 24);
}

static KSIMD_KERNEL_FORCE_INLINE ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt16_soft(uint16_t x)
{
    x = x - ((x >> 1) & UINT16_C(0x5555));
    x = (x & UINT16_C(0x3333)) + ((x >> 2) & UINT16_C(0x3333));
    x = (x + (x >> 4)) & UINT16_C(0x0f0f);
    return (ks_pop_bitcount_t)((x + (x >> 8)) & UINT16_C(0x001f));
}

static KSIMD_KERNEL_FORCE_INLINE ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt8_soft(uint8_t x)
{
    x = (x & UINT8_C(0x55)) + ((x >> 1) & UINT8_C(0x55));
    x = (x & UINT8_C(0x33)) + ((x >> 2) & UINT8_C(0x33));
    return (ks_pop_bitcount_t)((x + (x >> 4)) & UINT8_C(0x0f));
}

/* kernel */
KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_popcnt_buffer(const void* buffer, ks_bytesize_t byte_size);


/* for testing */
#ifdef KSIMD_IS_TESTING

KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, ks_bytesize_t byte_size);
KSIMD_KERNEL_POPCNT_API ks_pop_bitcount_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86_popcnt(const void* buffer, ks_bytesize_t byte_size);

#endif

KSIMD_KERNEL_END_EXTERN_C

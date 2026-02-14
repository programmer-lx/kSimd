#pragma once

/*
#define KSIMD_KERNEL_POPCNT_AS_DLL to use this kernel as DLL.

To calculate 1bit count in a buffer:
size_t count = ks_popcnt_buffer(&buffer, byte size of buffer);
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

static inline size_t KSIMD_KERNEL_CALL_CONV ks_popcnt64_soft(uint64_t x)
{
    uint64_t m1 = UINT64_C(0x5555555555555555);
    uint64_t m2 = UINT64_C(0x3333333333333333);
    uint64_t m4 = UINT64_C(0x0f0f0f0f0f0f0f0f);
    uint64_t h01 = UINT64_C(0x0101010101010101);

    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;

    return (size_t)((x * h01) >> 56);
}

static inline size_t KSIMD_KERNEL_CALL_CONV ks_popcnt8_soft(uint8_t x) {
    x = (x & 0x55) + ((x >> 1) & 0x55); // 每 2 位一组：01 01 01 01
    x = (x & 0x33) + ((x >> 2) & 0x33); // 每 4 位一组：00 11 00 11
    return (size_t)((x + (x >> 4)) & 0x0F); // 最后合并高 4 位和低 4 位
}

typedef size_t (KSIMD_KERNEL_CALL_CONV *ks_pfn_popcnt_buffer_t)(const void* buffer, size_t byte_size);

extern KSIMD_KERNEL_POPCNT_API ks_pfn_popcnt_buffer_t ks_popcnt_buffer;


/* for testing */
#ifdef KSIMD_KERNEL_IS_TESTING

KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_soft(const void* buffer, size_t byte_size);
KSIMD_KERNEL_POPCNT_API size_t KSIMD_KERNEL_CALL_CONV ks_test_popcnt_buffer_x86_popcnt(const void* buffer, size_t byte_size);

#endif

KSIMD_KERNEL_END_EXTERN_C

#pragma once

#ifdef __cplusplus
    #include <cstdint>
    #include <cstddef>
#else
    #include <stdint.h>
    #include <stddef.h>
#endif

#include "kSimd/kernels/kernel_api.h"

#ifdef KSIMD_KERNEL_CRC32C_AS_DLL
    /* shared lib */
    #ifdef KSIMD_KERNEL_CRC32C_EXPORT /* cpp file should define this macro. */
        #define KSIMD_KERNEL_CRC32C_API KSIMD_KERNEL_DLL_EXPORT
    #else
        #define KSIMD_KERNEL_CRC32C_API KSIMD_KERNEL_DLL_IMPORT
    #endif
#else
    /* static lib */
    #define KSIMD_KERNEL_CRC32C_API
#endif

KSIMD_KERNEL_BEGIN_EXTERN_C

#define ks_begin_crc32c() (UINT32_C(0xffffffff))
#define ks_end_crc32c(crc) ((crc) ^ UINT32_C(0xffffffff))

KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_update_crc32c(
    uint32_t origin,
    const uint8_t* data,
    size_t size
);


/* for testing */
#ifdef KSIMD_KERNEL_IS_TESTING
KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_test_update_crc32c_soft(
    uint32_t origin,
    const uint8_t* data,
    size_t size
);

KSIMD_KERNEL_CRC32C_API uint32_t KSIMD_KERNEL_CALL_CONV ks_test_update_crc32c_sse42(
    uint32_t origin,
    const uint8_t* data,
    size_t size
);
#endif

KSIMD_KERNEL_END_EXTERN_C

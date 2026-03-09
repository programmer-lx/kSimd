#pragma once

#ifdef __cplusplus
    #include <cstdint>
#else
    #include <stdint.h>
#endif

#include "kSimd/macros.h"

/*
kernel function must be pure C function.
*/

/* calling convention */
/* use __stdcall on windows, default on other OS. */
#if defined(_WIN32) || defined(_WIN64)
    #define KSIMD_KERNEL_CALL_CONV __stdcall
#else
    #define KSIMD_KERNEL_CALL_CONV
#endif


/* C export block */
#ifdef __cplusplus
    #define KSIMD_KERNEL_BEGIN_EXTERN_C extern "C" {
    #define KSIMD_KERNEL_END_EXTERN_C }
#else
    #define KSIMD_KERNEL_BEGIN_EXTERN_C
    #define KSIMD_KERNEL_END_EXTERN_C
#endif


/* fixed type define: 这些类型不随着平台的变化而变化，确保导出的ABI稳定 */
typedef uint64_t ks_bytesize_t;


/* macro for unit test */
/* #ifdef KSIMD_IS_TESTING */

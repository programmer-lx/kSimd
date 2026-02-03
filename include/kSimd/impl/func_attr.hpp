#pragma once

#include "platform.hpp"

// see https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html#x86-Function-Attributes for more intrinsics attributes information

// scalar
#define KSIMD_SCALAR_INTRINSIC_ATTR
#define KSIMD_OP_SCALAR_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SCALAR_INTRINSIC_ATTR


// sse
#define KSIMD_SSE_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse")
#define KSIMD_OP_SSE_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSE_INTRINSIC_ATTR


// sse2
#define KSIMD_SSE2_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse2")
#define KSIMD_OP_SSE2_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSE2_INTRINSIC_ATTR



// sse3
#define KSIMD_SSE3_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse3")
#define KSIMD_OP_SSE3_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSE3_INTRINSIC_ATTR

// ssse3
#define KSIMD_SSSE3_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("ssse3")
#define KSIMD_OP_SSSE3_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSSE3_INTRINSIC_ATTR


// sse4.1
#define KSIMD_SSE4_1_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse4.1")
#define KSIMD_OP_SSE4_1_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSE4_1_INTRINSIC_ATTR


// sse4.2
#define KSIMD_SSE4_2_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse4.2")
#define KSIMD_OP_SSE4_2_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SSE4_2_INTRINSIC_ATTR



// avx(no fma3)
#define KSIMD_AVX_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx")
#define KSIMD_OP_AVX_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_AVX_INTRINSIC_ATTR


// avx2(no fma3)
#define KSIMD_AVX2_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2")
#define KSIMD_OP_AVX2_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_AVX2_INTRINSIC_ATTR


// avx2+fma3+f16c
#define KSIMD_AVX2_FMA3_F16C_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2,fma,f16c")
#define KSIMD_OP_AVX2_FMA3_F16C_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_AVX2_FMA3_F16C_INTRINSIC_ATTR


// func sig
#define KSIMD_OP_SIG_SCALAR_STATIC(ret, func_name, params)         KSIMD_OP_SCALAR_API         static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_SCALAR(ret, func_name, params)                KSIMD_OP_SCALAR_API                ret KSIMD_CALL_CONV  func_name params noexcept

#define KSIMD_OP_SIG_SSE_STATIC(ret, func_name, params)            KSIMD_OP_SSE_API            static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_SSE(ret, func_name, params)                   KSIMD_OP_SSE_API                   ret KSIMD_CALL_CONV  func_name params noexcept

#define KSIMD_OP_SIG_SSE2_STATIC(ret, func_name, params)           KSIMD_OP_SSE2_API           static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_SSE2(ret, func_name, params)                  KSIMD_OP_SSE2_API                  ret KSIMD_CALL_CONV  func_name params noexcept

#define KSIMD_OP_SIG_SSE3_STATIC(ret, func_name, params)           KSIMD_OP_SSE3_API           static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_SSE4_1_STATIC(ret, func_name, params)         KSIMD_OP_SSE4_1_API         static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_SSE4_2_STATIC(ret, func_name, params)         KSIMD_OP_SSE4_2_API         static ret KSIMD_CALL_CONV  func_name params noexcept

#define KSIMD_OP_SIG_AVX(ret, func_name, params)                   KSIMD_OP_AVX_API                   ret KSIMD_CALL_CONV  func_name params noexcept

#define KSIMD_OP_SIG_AVX2_STATIC(ret, func_name, params)           KSIMD_OP_AVX2_API           static ret KSIMD_CALL_CONV  func_name params noexcept
#define KSIMD_OP_SIG_AVX2_FMA3_F16C_STATIC(ret, func_name, params) KSIMD_OP_AVX2_FMA3_F16C_API static ret KSIMD_CALL_CONV  func_name params noexcept

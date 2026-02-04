#pragma once

/*

KSIMD_INSTRUCTION_FEATURE 宏:
1. 根据CPU架构，预定义该CPU可能支持的SIMD指令，比如我定义了 KSIMD_INSTRUCTION_FEATURE_SSE ，代表我假设目标平台支持SSE指令
2. 函数分发逻辑(分发表的构建、分发索引的选择...)只能依赖 INSTRUCTION_FEATURE 宏，不能依赖任何的 ARCH 宏。
    这样就可以做到精确的 **指令级别** 的分发，而不是根据平台分发。(比如可以强制注释掉某个 FEATURE 宏，就可以一键禁用某个指令的分发)
    当然，未来可以添加一些 cmake option，在保证库的基线指令被分发的前提下，让开发者决定额外分发某些指令
*/

#include "common_macros.hpp"

// compiler
#if !defined(KSIMD_COMPILER_MSVC) && !defined(KSIMD_COMPILER_GCC) && !defined(KSIMD_COMPILER_CLANG)
    #error "Unknown compiler, only support MSVC, GCC, clang"
#endif


// --- X86 系列 ---
// ----------------------------- x86 64-bit -----------------------------
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define KSIMD_ARCH_X86_64
    #define KSIMD_ARCH_X86_ANY
// ----------------------------- x86 32-bit -----------------------------
#elif defined(_M_IX86) || defined(__i386__)
    #define KSIMD_ARCH_X86_32
    #define KSIMD_ARCH_X86_ANY
#else
    #error "Unknown arch, tSimd can only support x86 arch."
#endif



// SIMD support
// AVX512 -> AVX2 -> FMA3(独立的指令开关) -> AVX -> SSE4.1(不一定有SSE4.2) -> SSE3 -> SSE2
//                -> F16C(独立的指令开关) -> AVX
// SSE4.2 -> SSE4.1
// x86 64 -> SSE2


// call conv
#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64) && !defined(_M_HYBRID_X86_ARM64) && !defined(_M_ARM64EC) && (!_MANAGED) && (!_M_CEE) && (!defined(_M_IX86_FP) || (_M_IX86_FP > 1)) && !defined(KSIMD_VECTORCALL_ENABLED)
    #define KSIMD_VECTORCALL_ENABLED
#endif

#if defined(KSIMD_VECTORCALL_ENABLED)
    #define KSIMD_CALL_CONV __vectorcall
#else
    #define KSIMD_CALL_CONV
#endif


// ------------------------------------------- instruction features -------------------------------------------
// 这些宏开关，表示分发表将会分发哪些函数
// fallback指令的值，后续可通过类似于
// #if (KSIMD_INSTRUCTION_FEATURE_SSE == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE) 的判断，来判断这个指令是不是fallback
#define KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE (-1) // fallback值
#undef KSIMD_DETAIL_INST_FEATURE_FALLBACK

// Scalar
#if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_ANY)
    #define KSIMD_INSTRUCTION_FEATURE_SCALAR KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #define KSIMD_DETAIL_INST_FEATURE_FALLBACK // fallback
#endif

// --------- x86指令集 ---------
#if defined(KSIMD_ARCH_X86_ANY)

    #define KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY
    // SSE: 只在 x86 32bit 提供SSE分发
    #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_32)
        // #define KSIMD_INSTRUCTION_FEATURE_SSE 1      // 2026 无需支持
    #endif

    // SSE2 及以上
    #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_ANY)
        // SSE2
        // #define KSIMD_INSTRUCTION_FEATURE_SSE2 1     // 2026 无需支持
        // // SSE2: fallback 1
        // #if !defined(KSIMD_DETAIL_INST_FEATURE_FALLBACK)
        //     #undef KSIMD_INSTRUCTION_FEATURE_SSE2
        //     #define KSIMD_INSTRUCTION_FEATURE_SSE2 KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE // fallback value
        //     #define KSIMD_DETAIL_INST_FEATURE_FALLBACK // fallback
        // #endif

        // #define KSIMD_INSTRUCTION_FEATURE_SSE3 1     // 2026 无需支持
        // #define KSIMD_INSTRUCTION_FEATURE_SSSE3 1    // 2026 无需支持
        #define KSIMD_INSTRUCTION_FEATURE_SSE4_1 1
        // #define KSIMD_INSTRUCTION_FEATURE_SSE4_2 1
    #endif

    // AVX family
    #define KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY
    #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_ANY)
        // #define KSIMD_INSTRUCTION_FEATURE_AVX 1      // 2026 无需支持
        // #define KSIMD_INSTRUCTION_FEATURE_AVX2 1     // 2026 无需支持
        #define KSIMD_INSTRUCTION_FEATURE_AVX2_FMA3 1
    #endif

    // AVX-512 family
    #define KSIMD_INSTRUCTION_FEATURE_AVX512_FAMILY
    #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_ANY)
        // #define KSIMD_INSTRUCTION_FEATURE_AVX512_F 1
    #endif

#endif // x86 instructions

// check fallback
#if !defined(KSIMD_DETAIL_INST_FEATURE_FALLBACK)
    #error "we must define a fallback instruction."
#endif

KSIMD_NAMESPACE_BEGIN

namespace alignment
{
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec128 = 16;
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec256 = 32;
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec512 = 64;
}

struct CpuSupportInfo
{
    static constexpr unsigned Scalar = 1;

    unsigned FXSR       : 1 = 0;

    // SSE family
    unsigned SSE        : 1 = 0;
    unsigned SSE2       : 1 = 0;
    unsigned SSE3       : 1 = 0;
    unsigned SSSE3      : 1 = 0;
    unsigned SSE4_1     : 1 = 0;
    unsigned SSE4_2     : 1 = 0;

    // XSAVE & OS_XSAVE
    unsigned XSAVE      : 1 = 0;
    unsigned OS_XSAVE   : 1 = 0;

    // AVX family
    unsigned AVX        : 1 = 0;

    // 这两个是独立指令集，在tsimd库中，AVX的op不使用FMA3+F16C指令，AVX2的op分成两套:
    // 1. AVX2, 2. AVX2+FMA3+F16C。一套不使用FMA3+F16C，另一套使用FMA3+F16C
    unsigned F16C       : 1 = 0;
    unsigned FMA3       : 1 = 0;

    unsigned AVX2       : 1 = 0;

    // AVX-512 family
    unsigned AVX512_F   : 1 = 0; // AVX512F支持FMA运算，不需要单独划分FMA3支持
};

CpuSupportInfo get_cpu_support_info() noexcept;

KSIMD_NAMESPACE_END

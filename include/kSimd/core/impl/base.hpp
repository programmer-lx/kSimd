#pragma once

// compiler detect
#if defined(_MSC_VER) && !defined(__clang__)
    #define KSIMD_COMPILER_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
    #define KSIMD_COMPILER_GCC 1
#elif defined(__clang__)
    #define KSIMD_COMPILER_CLANG 1
#endif

#ifndef __cplusplus
    #error "requires C++."
#endif
#if !KSIMD_COMPILER_MSVC && !KSIMD_COMPILER_GCC && !KSIMD_COMPILER_CLANG
    #error "Unknown compiler, only support msvc, g++, clang++."
#endif

#define KSIMD_STR_IMPL(x) #x
#define KSIMD_STR(x) KSIMD_STR_IMPL(x)

#define KSIMD_CONCAT_IMPL(a, b) a##b
#define KSIMD_CONCAT(a, b) KSIMD_CONCAT_IMPL(a, b)

#define KSIMD_IGNORE_WARNING_MSVC(warnings)
#define KSIMD_IGNORE_WARNING_GCC(warnings)
#define KSIMD_IGNORE_WARNING_CLANG(warnings)

#if defined(KSIMD_COMPILER_MSVC)

    #define KSIMD_RESTRICT __restrict
    #define KSIMD_NOINLINE __declspec(noinline)
    #define KSIMD_FORCE_INLINE __forceinline
    #define KSIMD_FLATTEN
    #define KSIMD_PRAGMA(tokens) __pragma(tokens)

    #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(warning(push))
    #define KSIMD_WARNING_POP KSIMD_PRAGMA(warning(pop))
    #undef KSIMD_IGNORE_WARNING_MSVC
    #define KSIMD_IGNORE_WARNING_MSVC(warnings) KSIMD_PRAGMA(warning(disable : warnings))

    #define KSIMD_FUNC_ATTR_INTRINSIC_TARGETS(...)

#elif defined(KSIMD_COMPILER_GCC) || defined(KSIMD_COMPILER_CLANG) // GCC clang

    #define KSIMD_RESTRICT __restrict__
    #define KSIMD_NOINLINE __attribute__((noinline))
    #define KSIMD_FORCE_INLINE inline __attribute__((always_inline))
    #define KSIMD_FLATTEN __attribute__((flatten))
    #define KSIMD_PRAGMA(tokens) _Pragma(#tokens)
    #define KSIMD_FUNC_ATTR_INTRINSIC_TARGETS(...) __attribute__((target(__VA_ARGS__)))

    #if !defined(KSIMD_COMPILER_CLANG) // GCC only

        #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(GCC diagnostic push)
        #define KSIMD_WARNING_POP KSIMD_PRAGMA(GCC diagnostic pop)
        #undef KSIMD_IGNORE_WARNING_GCC
        #define KSIMD_IGNORE_WARNING_GCC(warnings) KSIMD_PRAGMA(GCC diagnostic ignored warnings)

    #endif

    #if !defined(KSIMD_COMPILER_GCC) // clang only

        #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(clang diagnostic push)
        #define KSIMD_WARNING_POP KSIMD_PRAGMA(clang diagnostic pop)
        #undef KSIMD_IGNORE_WARNING_CLANG
        #define KSIMD_IGNORE_WARNING_CLANG(warnings) KSIMD_PRAGMA(clang diagnostic ignored warnings)

    #endif

#endif // MSVC

// Header-only 全局常量或 constexpr 函数 (防止误用 static constexpr 导致每个TU一份)
#define KSIMD_HEADER_GLOBAL_CONSTEXPR inline constexpr

// Header-only 全局变量或内联函数 (防止误用 static)
#define KSIMD_HEADER_GLOBAL inline


// std::floatXXX support
#ifdef __STDCPP_FLOAT16_T__
    #define KSIMD_SUPPORT_STD_FLOAT16 1
#endif

#ifdef __STDCPP_BFLOAT16_T__
    #define KSIMD_SUPPORT_STD_BFLOAT16 1
#endif

#ifdef __STDCPP_FLOAT32_T__
    #define KSIMD_SUPPORT_STD_FLOAT32 1
#endif

#ifdef __STDCPP_FLOAT64_T__
    #define KSIMD_SUPPORT_STD_FLOAT64 1
#endif


// --- X86 系列 ---
// ----------------------------- x86 64-bit -----------------------------
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define KSIMD_ARCH_X86_64 1
    #define KSIMD_ARCH_X86_ANY 1
// ----------------------------- x86 32-bit -----------------------------
#elif defined(_M_IX86) || defined(__i386__)
    #define KSIMD_ARCH_X86_32 1
    #define KSIMD_ARCH_X86_ANY 1
#else
    #error "Unknown arch, kSimd can only support x86 arch."
#endif


// call conv
#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64) && !defined(_M_HYBRID_X86_ARM64) && !defined(_M_ARM64EC) && (!_MANAGED) && (!_M_CEE) && (!defined(_M_IX86_FP) || (_M_IX86_FP > 1)) && !defined(KSIMD_VECTORCALL_ENABLED)
    #define KSIMD_VECTORCALL_ENABLED 1
#endif

#if KSIMD_VECTORCALL_ENABLED
    #define KSIMD_CALL_CONV __vectorcall
#else
    #define KSIMD_CALL_CONV
#endif


// ------------------------------------------- instruction features -------------------------------------------
// 这些宏开关，表示分发表将会分发哪些函数
// fallback指令的值，后续可通过类似于
#define KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE (-1) // fallback值
#undef KSIMD_DETAIL_INST_FEATURE_FALLBACK

// Scalar
#if KSIMD_IS_TESTING || KSIMD_ARCH_X86_ANY
    #define KSIMD_INSTRUCTION_FEATURE_SCALAR KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #define KSIMD_DETAIL_INST_FEATURE_FALLBACK 1 // fallback
#endif

// --------- x86指令集 ---------
#if KSIMD_ARCH_X86_ANY

    // #define KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY      // 2026 无需支持
    // SSE: 只在 x86 32bit 提供SSE分发
    // #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_32)
        // #define KSIMD_INSTRUCTION_FEATURE_SSE 1      // 2026 无需支持
    // #endif

    // SSE2 及以上
    // #if defined(KSIMD_IS_TESTING) || defined(KSIMD_ARCH_X86_ANY)
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
        // #define KSIMD_INSTRUCTION_FEATURE_SSE4_1 1   // 2026 无需支持
        // #define KSIMD_INSTRUCTION_FEATURE_SSE4_2 1
    // #endif

    // AVX family
    #define KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY
    #if KSIMD_IS_TESTING || KSIMD_ARCH_X86_ANY
        // #define KSIMD_INSTRUCTION_FEATURE_AVX 1      // 2026 无需支持
        // #define KSIMD_INSTRUCTION_FEATURE_AVX2 1     // 2026 无需支持
        #define KSIMD_INSTRUCTION_FEATURE_AVX2_MAX 1    // AVX2+FMA3+F16C
    #endif

    // AVX-512 family
    #define KSIMD_INSTRUCTION_FEATURE_AVX512_FAMILY
    #if KSIMD_IS_TESTING || KSIMD_ARCH_X86_ANY
        // #define KSIMD_INSTRUCTION_FEATURE_AVX512_F 1
    #endif

#endif // x86 instructions

// check fallback
#if !KSIMD_DETAIL_INST_FEATURE_FALLBACK
    #error "we must define a fallback instruction."
#endif

namespace ksimd
{
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
        unsigned F16C       : 1 = 0;
        unsigned FMA3       : 1 = 0;
        unsigned AVX2       : 1 = 0;

        // AVX-512 family
        unsigned AVX512_F   : 1 = 0; // AVX512F支持FMA运算，不需要单独划分FMA3支持
    };

    const CpuSupportInfo& get_cpu_support_info() noexcept;
}

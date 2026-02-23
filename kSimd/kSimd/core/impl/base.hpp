#pragma once

#include "kSimd/macros.h"

#include <type_traits>
#include <version>

#include <cstdint>


// C++ standard
#ifndef __cplusplus
    #error requires C++.
#endif

// bit_cast
#if __cpp_lib_bit_cast < 201806L
    #error requires std::bit_cast
#endif

// concept
#if __cpp_concepts < 201907L
    #error requires concepts.
#endif

// lambda template
#if __cpp_generic_lambdas < 201707L
    #error requires generic lambdas.
#endif

// hex float
#if __cpp_hex_float >= 201603L
    #define KSIMD_SUPPORT_STD_HEX_FLOAT 1
#endif

// C++23 std::floatXXX support
#if __has_include(<stdfloat>)
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
#endif


// simd inline op 的调用约定
#if defined(_MSC_VER) && !defined(_M_ARM) && !defined(_M_ARM64) && !defined(_M_HYBRID_X86_ARM64) && \
    !defined(_M_ARM64EC) && (!_MANAGED) && (!_M_CEE) && (!defined(_M_IX86_FP) || (_M_IX86_FP > 1))

    #define KSIMD_CALL_CONV __vectorcall
#else
    #define KSIMD_CALL_CONV
#endif

// min, max macro
#if defined(min) || defined(max)
    #error The "min" "max" macro are defined, we can define NOMINMAX before include <windows.h>.
#endif

// Header-only 全局常量或 constexpr 函数 (防止误用 static constexpr 导致每个TU一份)
#define KSIMD_HEADER_GLOBAL_CONSTEXPR inline constexpr

// Header-only 全局变量或内联函数 (防止误用 static)
#define KSIMD_HEADER_GLOBAL inline


// ------------------------------------------- instruction features -------------------------------------------
// 可通过定义 KSIMD_DISABLE_XXX 来取消某些路径的分发
// 要在包含 kSimd 的文件之前，使用这种方式定义宏:
// #undef KSIMD_DISABLE_AVX2_MAX
// #define KSIMD_DISABLE_AVX2_MAX // 因为文件会被多次包含，所以需要不断的取消定义再重新定义
//
// #undef KSIMD_DISPATCH_THIS_FILE
// #define KSIMD_DISPATCH_THIS_FILE "XXX"
// #include <kSimd/core/dispatch_this_file.hpp>
// #include <kSimd/core/dispatch_core.hpp>

// 目前支持的宏:
// - KSIMD_DISABLE_AVX2_MAX: 取消 AVX2_FMA3_F16C 的分发

// KSIMD_DISABLE_XXX 系列宏，用户定制分发上限
// 而下面由编译器定义的宏，比如__AVX2__，用来定义分发表的下限

// 在编译期进行分发表裁剪，如果已经打开了AVX2+FMA3+F16C开关，那么其实就没必要分发标量了
#if KSIMD_COMPILER_MSVC
    #ifdef __AVX2__
        #define KSIMD_BASELINE_AVX2_MAX 1
    #endif
#elif KSIMD_COMPILER_GCC || KSIMD_COMPILER_CLANG
    #if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
        #define KSIMD_BASELINE_AVX2_MAX 1
    #endif
#endif

// 这些宏开关，表示分发表将会分发哪些函数
// fallback指令的值，后续可通过类似于
#define KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE (-1) // fallback值
#undef KSIMD_DETAIL_INST_FEATURE_FALLBACK

// 由高到低判断

// --------- x86指令集 ---------
#if KSIMD_ARCH_X86_ANY
    // AVX2+FMA3+F16C (AVX2_MAX)
    #if defined(KSIMD_IS_TESTING) || (!defined(KSIMD_DISABLE_AVX2_MAX) && !KSIMD_DETAIL_INST_FEATURE_FALLBACK)
        #define KSIMD_INSTRUCTION_FEATURE_AVX2_MAX 1

        #if KSIMD_BASELINE_AVX2_MAX
            #undef KSIMD_INSTRUCTION_FEATURE_AVX2_MAX
            #define KSIMD_INSTRUCTION_FEATURE_AVX2_MAX KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
            #define KSIMD_DETAIL_INST_FEATURE_FALLBACK 1 // mark as fallback instruction
        #endif
    #endif

#endif // x86 instructions

// --------- arm指令集 ---------
#if KSIMD_ARCH_ARM_ANY
    // NEON
    #if defined(KSIMD_IS_TESTING) || (!KSIMD_DETAIL_INST_FEATURE_FALLBACK)
        #define KSIMD_INSTRUCTION_FEATURE_NEON 1
    #endif
#endif // arm instructions

// Scalar
#if defined(KSIMD_IS_TESTING) || (!KSIMD_DETAIL_INST_FEATURE_FALLBACK)
    #define KSIMD_INSTRUCTION_FEATURE_SCALAR KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #define KSIMD_DETAIL_INST_FEATURE_FALLBACK 1 // fallback
#endif

// check fallback
#if !KSIMD_DETAIL_INST_FEATURE_FALLBACK
    #error "we must define a fallback instruction."
#endif

namespace ksimd
{
    enum class CpuVendor
    {
        Unknown = 0,
        Intel,
        AMD
    };

    struct CpuSupportInfo
    {
        // ------------------ common info ------------------
        CpuVendor vendor                    = CpuVendor::Unknown;
        char      vendor_name[13]           = {};
        unsigned  logical_cores             = 0;
        unsigned  physical_cores            = 0;

        unsigned hyper_threads          : 1 = 0;

        // ------------------ x86 features ------------------
        unsigned fxsr                   : 1 = 0;

        // SSE family
        unsigned sse                    : 1 = 0;
        unsigned sse2                   : 1 = 0;
        unsigned sse3                   : 1 = 0;
        unsigned ssse3                  : 1 = 0;
        unsigned sse4_1                 : 1 = 0;
        unsigned sse4_2                 : 1 = 0;

        // XSAVE & OS_XSAVE
        unsigned xsave                  : 1 = 0;
        unsigned os_xsave               : 1 = 0;

        // AVX family
        unsigned avx                    : 1 = 0;
        unsigned f16c                   : 1 = 0;
        unsigned fma3                   : 1 = 0;
        unsigned avx2                   : 1 = 0;
        unsigned avx_vnni               : 1 = 0;
        unsigned avx_vnni_int8          : 1 = 0;
        unsigned avx_ne_convert         : 1 = 0;
        unsigned avx_ifma               : 1 = 0;
        unsigned avx_vnni_int16         : 1 = 0;
        unsigned sha512                 : 1 = 0;
        unsigned sm3                    : 1 = 0;
        unsigned sm4                    : 1 = 0;

        // AVX-512 family
        unsigned avx512_f               : 1 = 0;
        unsigned avx512_bw              : 1 = 0;
        unsigned avx512_cd              : 1 = 0;
        unsigned avx512_dq              : 1 = 0;
        unsigned avx512_ifma            : 1 = 0;
        unsigned avx512_vl              : 1 = 0;
        unsigned avx512_vpopcntdq       : 1 = 0;
        unsigned avx512_bf16            : 1 = 0;
        unsigned avx512_bitalg          : 1 = 0;
        unsigned avx512_vbmi            : 1 = 0;
        unsigned avx512_vbmi2           : 1 = 0;
        unsigned avx512_vnni            : 1 = 0;
        unsigned avx512_vp2intersect    : 1 = 0;
        unsigned avx512_fp16            : 1 = 0;

        // other
        unsigned popcnt                 : 1 = 0;
        unsigned aes_ni                 : 1 = 0;
        unsigned sha                    : 1 = 0;

        // ------------------ arm features ------------------
        unsigned neon                   : 1 = 0;
        unsigned sve                    : 1 = 0;
        unsigned arm_crc32              : 1 = 0;
    };

    namespace detail
    {
        template<typename T>
        using underlying_t =
            std::conditional_t<
                std::is_enum_v<T>,
                std::underlying_type_t<T>,
                T
            >;

        template<typename T>
            requires (std::is_enum_v<T> || std::is_integral_v<T>)
        constexpr underlying_t<T> underlying(const T val) noexcept
        {
            return static_cast<underlying_t<T>>(val);
        }
    }

    const CpuSupportInfo& get_cpu_support_info() noexcept;
}

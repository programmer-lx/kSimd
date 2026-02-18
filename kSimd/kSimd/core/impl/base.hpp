#pragma once

#include <cstdint>

#include <type_traits>
#include <version>

// compiler detect
#if defined(_MSC_VER) && !defined(__clang__)
    #define KSIMD_COMPILER_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
    #define KSIMD_COMPILER_GCC 1
#elif defined(__clang__)
    #define KSIMD_COMPILER_CLANG 1
#endif

#if defined(KSIMD_COMPILER_MSVC) + defined(KSIMD_COMPILER_GCC) + defined(KSIMD_COMPILER_CLANG) != 1
    #error "We should only define one compiler macro."
#endif

#if !KSIMD_COMPILER_MSVC && !KSIMD_COMPILER_GCC && !KSIMD_COMPILER_CLANG
    #error "Unknown compiler, only support msvc, g++, clang++."
#endif


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


// OS detect
#if defined(_WIN32) || defined(_WIN64)
    #define KSIMD_OS_WINDOWS 1
#endif

#if defined(__APPLE__) && defined(__MACH__)
    #define KSIMD_OS_MACOS 1
#endif

#if defined(__linux__)
    #define KSIMD_OS_LINUX 1
#endif


// arch
// ----------------------------- x86 64-bit -----------------------------
#if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define KSIMD_ARCH_X86_64 1
    #define KSIMD_ARCH_X86_ANY 1

// ----------------------------- x86 32-bit -----------------------------
#elif defined(_M_IX86) || defined(__i386__)
    #define KSIMD_ARCH_X86_32 1
    #define KSIMD_ARCH_X86_ANY 1

// ----------------------------- ARM 64-bit -----------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define KSIMD_ARCH_ARM_64 1
    #define KSIMD_ARCH_ARM_ANY 1

// ----------------------------- ARM 32-bit -----------------------------
#elif defined(__arm__) || defined(_M_ARM) || defined(__arm64_32__)
    #define KSIMD_ARCH_ARM_32 1
    #define KSIMD_ARCH_ARM_ANY 1

#else
    #error "Unknown arch, kSimd can only support x86 and arm."
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

// --- macro utils ---
#define KSIMD_STR_IMPL(x) #x
#define KSIMD_STR(x) KSIMD_STR_IMPL(x)

#define KSIMD_CONCAT_IMPL(a, b) a##b
#define KSIMD_CONCAT(a, b) KSIMD_CONCAT_IMPL(a, b)

#define KSIMD_IGNORE_WARNING_MSVC(warnings)
#define KSIMD_IGNORE_WARNING_GCC(warnings)
#define KSIMD_IGNORE_WARNING_CLANG(warnings)

#if KSIMD_COMPILER_MSVC

    #define KSIMD_DLL_LOCAL
    #define KSIMD_DLL_IMPORT __declspec(dllexport)
    #define KSIMD_DLL_EXPORT __declspec(dllimport)

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

#elif KSIMD_COMPILER_GCC || KSIMD_COMPILER_CLANG // GCC clang

    #define KSIMD_DLL_LOCAL  __attribute__((visibility("hidden")))
    #define KSIMD_DLL_IMPORT __attribute__((visibility("default")))
    #define KSIMD_DLL_EXPORT __attribute__((visibility("default")))

    #define KSIMD_RESTRICT __restrict__
    #define KSIMD_NOINLINE __attribute__((noinline))
    #define KSIMD_FORCE_INLINE inline __attribute__((always_inline))
    #define KSIMD_FLATTEN __attribute__((flatten))
    #define KSIMD_PRAGMA(tokens) _Pragma(#tokens)
    #define KSIMD_FUNC_ATTR_INTRINSIC_TARGETS(...) __attribute__((target(__VA_ARGS__)))

    #if !KSIMD_COMPILER_CLANG // GCC only

        #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(GCC diagnostic push)
        #define KSIMD_WARNING_POP KSIMD_PRAGMA(GCC diagnostic pop)
        #undef KSIMD_IGNORE_WARNING_GCC
        #define KSIMD_IGNORE_WARNING_GCC(warnings) KSIMD_PRAGMA(GCC diagnostic ignored warnings)

    #endif

    #if !KSIMD_COMPILER_GCC // clang only

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
    struct CpuSupportInfo
    {
        static constexpr unsigned Scalar = 1;

        // --- x86 ---
        unsigned POPCNT     : 1 = 0;
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
        unsigned AVX512_F   : 1 = 0;

        // --- arm ---
        unsigned NEON       : 1 = 0;
        unsigned SVE        : 1 = 0;
        unsigned ARM_CRC32  : 1 = 0;
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

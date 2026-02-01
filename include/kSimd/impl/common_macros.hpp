#pragma once

#include <cstddef>
#include <cassert>
#include <cstdlib>

#define KSIMD_NAMESPACE_NAME ksimd
#define KSIMD_NAMESPACE_BEGIN namespace KSIMD_NAMESPACE_NAME {
#define KSIMD_NAMESPACE_END }


// compiler detect
#if defined(_MSC_VER) && !defined(__clang__)
    #define KSIMD_COMPILER_MSVC
#elif defined(__GNUC__) && !defined(__clang__)
    #define KSIMD_COMPILER_GCC
#elif defined(__clang__)
    #define KSIMD_COMPILER_CLANG
#endif

#ifndef __cplusplus
    #error "requires C++."
#endif
#if !defined(KSIMD_COMPILER_MSVC) && !defined(KSIMD_COMPILER_GCC) && !defined(KSIMD_COMPILER_CLANG)
    #error "Unknown compiler, only support msvc, g++, clang++."
#endif

#define KSIMD_STR_IMPL(x) #x
#define KSIMD_STR(x) KSIMD_STR_IMPL(x)

#define KSIMD_CONCAT_IMPL(a, b) a##b
#define KSIMD_CONCAT(a, b) KSIMD_CONCAT_IMPL(a, b)

#define KSIMD_NO_DISCARD [[nodiscard]]
#define KSIMD_NORETURN [[noreturn]]

#define KSIMD_ASSERT(...) assert(__VA_ARGS__);

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

// dll export
#if defined(_MSC_VER)
    #define KSIMD_API_EXPORT __declspec(dllexport)
    #define KSIMD_API_IMPORT __declspec(dllimport)
#else
    #define KSIMD_API_EXPORT __attribute__((visibility("default")))
    #define KSIMD_API_IMPORT __attribute__((visibility("default")))
#endif

// Header-only 全局常量或 constexpr 函数 (防止误用 static constexpr 导致每个TU一份)
#define KSIMD_HEADER_GLOBAL_CONSTEXPR inline constexpr

// Header-only 全局变量或内联函数 (防止误用 static)
#define KSIMD_HEADER_GLOBAL inline

#pragma once

// compiler detect
#define KSIMD_COMPILER_MSVC 0
#define KSIMD_COMPILER_GCC 0
#define KSIMD_COMPILER_CLANG_CL 0
#define KSIMD_COMPILER_CLANG 0

#if defined(__clang__)
    #if defined(_MSC_VER)
        #undef KSIMD_COMPILER_CLANG_CL
        #define KSIMD_COMPILER_CLANG_CL 1
    #else
        #undef KSIMD_COMPILER_CLANG
        #define KSIMD_COMPILER_CLANG 1
    #endif
#elif defined(__GNUC__)
    #undef KSIMD_COMPILER_GCC
    #define KSIMD_COMPILER_GCC 1
#elif defined(_MSC_VER)
    #undef KSIMD_COMPILER_MSVC
    #define KSIMD_COMPILER_MSVC 1
#endif

#if KSIMD_COMPILER_MSVC + KSIMD_COMPILER_GCC + KSIMD_COMPILER_CLANG + KSIMD_COMPILER_CLANG_CL != 1
    #error "We should only define one compiler macro."
#endif

#if !KSIMD_COMPILER_MSVC && !KSIMD_COMPILER_GCC && !KSIMD_COMPILER_CLANG && !KSIMD_COMPILER_CLANG_CL
    #error "Unknown compiler, only support msvc, g++, clang++, clang-cl."
#endif

// OS
#define KSIMD_OS_WINDOWS 0
#if defined(_WIN32) || defined(_WIN64)
    #undef KSIMD_OS_WINDOWS
    #define KSIMD_OS_WINDOWS 1
#endif

#define KSIMD_OS_MACOS 0
#if defined(__APPLE__) && defined(__MACH__)
    #undef KSIMD_OS_MACOS
    #define KSIMD_OS_MACOS 1
#endif

#define KSIMD_OS_LINUX 0
#if defined(__linux__)
    #undef KSIMD_OS_LINUX
    #define KSIMD_OS_LINUX 1
#endif

#define KSIMD_OS_ANDROID 0
#if defined(__ANDROID__)
    #undef KSIMD_OS_ANDROID
    #define KSIMD_OS_ANDROID 1
#endif

// arch
#define KSIMD_ARCH_X86_64 0
#define KSIMD_ARCH_X86_32 0
#define KSIMD_ARCH_X86_ANY 0
#define KSIMD_ARCH_ARM_64 0
#define KSIMD_ARCH_ARM_32 0
#define KSIMD_ARCH_ARM_ANY 0
// ----------------------------- x86 64-bit -----------------------------
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #undef KSIMD_ARCH_X86_64
    #define KSIMD_ARCH_X86_64 1

    #undef KSIMD_ARCH_X86_ANY
    #define KSIMD_ARCH_X86_ANY 1

// ----------------------------- x86 32-bit -----------------------------
#elif defined(_M_IX86) || defined(__i386__)
    #undef KSIMD_ARCH_X86_32
    #define KSIMD_ARCH_X86_32 1

    #undef KSIMD_ARCH_X86_ANY
    #define KSIMD_ARCH_X86_ANY 1
    #error x86 32bit is unsupported, please use x86 64bit

// ----------------------------- ARM 64-bit -----------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)
    #undef KSIMD_ARCH_ARM_64
    #define KSIMD_ARCH_ARM_64 1

    #undef KSIMD_ARCH_ARM_ANY
    #define KSIMD_ARCH_ARM_ANY 1

// ----------------------------- ARM 32-bit -----------------------------
#elif defined(__arm__) || defined(_M_ARM) || defined(__arm64_32__)
    #undef KSIMD_ARCH_ARM_32
    #define KSIMD_ARCH_ARM_32 1

    #undef KSIMD_ARCH_ARM_ANY
    #define KSIMD_ARCH_ARM_ANY 1
    #error arm 32bit is unsupported, please use arm 64bit

#else
    #error Unknown arch, kSimd can only support x86-64 and arm-64.
#endif


// Native extended floating-point type (such as x86 _Float16, arm __fp16)
// _Float16
#if defined(__FLT16_MAX__) || KSIMD_ARCH_ARM_ANY
    #define KSIMD_SUPPORT_EXTENSION_FLOAT16 1
#else
    #define KSIMD_SUPPORT_EXTENSION_FLOAT16 0
#endif

// _Float32
#if defined(__FLT32_MAX__)
    #define KSIMD_SUPPORT_EXTENSION_FLOAT32 1
#else
    #define KSIMD_SUPPORT_EXTENSION_FLOAT32 0
#endif

// _Float64
#if defined(__FLT64_MAX__)
    #define KSIMD_SUPPORT_EXTENSION_FLOAT64 1
#else
    #define KSIMD_SUPPORT_EXTENSION_FLOAT64 0
#endif

// _Float128
#if defined(__FLT128_MAX__)
    #define KSIMD_SUPPORT_EXTENSION_FLOAT128 1
#else
    #define KSIMD_SUPPORT_EXTENSION_FLOAT128 0
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

#elif KSIMD_COMPILER_GCC || KSIMD_COMPILER_CLANG || KSIMD_COMPILER_CLANG_CL // GCC clang

    #define KSIMD_DLL_LOCAL  __attribute__((visibility("hidden")))
    #define KSIMD_DLL_IMPORT __attribute__((visibility("default")))
    #define KSIMD_DLL_EXPORT __attribute__((visibility("default")))

    #define KSIMD_RESTRICT __restrict__
    #define KSIMD_NOINLINE __attribute__((noinline))
    #define KSIMD_FORCE_INLINE inline __attribute__((always_inline))
    #define KSIMD_FLATTEN __attribute__((flatten))
    #define KSIMD_PRAGMA(tokens) _Pragma(#tokens)
    #define KSIMD_FUNC_ATTR_INTRINSIC_TARGETS(...) __attribute__((target(__VA_ARGS__)))
#endif // MSVC

#if KSIMD_COMPILER_GCC // GCC only

    #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(GCC diagnostic push)
    #define KSIMD_WARNING_POP KSIMD_PRAGMA(GCC diagnostic pop)
    #undef KSIMD_IGNORE_WARNING_GCC
    #define KSIMD_IGNORE_WARNING_GCC(warnings) KSIMD_PRAGMA(GCC diagnostic ignored warnings)

#endif

#if KSIMD_COMPILER_CLANG || KSIMD_COMPILER_CLANG_CL // clang only

    #define KSIMD_WARNING_PUSH KSIMD_PRAGMA(clang diagnostic push)
    #define KSIMD_WARNING_POP KSIMD_PRAGMA(clang diagnostic pop)
    #undef KSIMD_IGNORE_WARNING_CLANG
    #define KSIMD_IGNORE_WARNING_CLANG(warnings) KSIMD_PRAGMA(clang diagnostic ignored warnings)

#endif
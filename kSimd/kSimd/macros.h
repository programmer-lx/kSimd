#pragma once

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

// OS
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
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__amd64__)
    #define KSIMD_ARCH_X86_64 1
    #define KSIMD_ARCH_X86_ANY 1

// ----------------------------- x86 32-bit -----------------------------
#elif defined(_M_IX86) || defined(__i386__)
    #define KSIMD_ARCH_X86_32 1
    #define KSIMD_ARCH_X86_ANY 1
    #error x86 32bit is unsupported, please use x86 64bit

// ----------------------------- ARM 64-bit -----------------------------
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define KSIMD_ARCH_ARM_64 1
    #define KSIMD_ARCH_ARM_ANY 1

// ----------------------------- ARM 32-bit -----------------------------
#elif defined(__arm__) || defined(_M_ARM) || defined(__arm64_32__)
    #define KSIMD_ARCH_ARM_32 1
    #define KSIMD_ARCH_ARM_ANY 1
    #error arm 32bit is unsupported, please use arm 64bit

#else
    #error "Unknown arch, kSimd can only support x86 and arm."
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

#define KSIMD_PRAGMA_MESSAGE(msg) KSIMD_PRAGMA(message("[kSimd] - " msg))

#pragma once

/*
kernel function must be pure C function.
*/

#if defined(_MSC_VER) && !defined(__clang__)
    #define KSIMD_KERNEL_COMPILER_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
    #define KSIMD_KERNEL_COMPILER_GCC 1
#elif defined(__clang__)
    #define KSIMD_KERNEL_COMPILER_CLANG 1
#endif

#if defined(KSIMD_KERNEL_COMPILER_MSVC) + defined(KSIMD_KERNEL_COMPILER_GCC) + defined(KSIMD_KERNEL_COMPILER_CLANG) != 1
    #error We should only define one compiler macro.
#endif

#if !KSIMD_KERNEL_COMPILER_MSVC && !KSIMD_KERNEL_COMPILER_GCC && !KSIMD_KERNEL_COMPILER_CLANG
    #error Unknown compiler, only support msvc, g++, clang++.
#endif


/* DLL import/export */
#if KSIMD_KERNEL_COMPILER_MSVC
    #define KSIMD_KERNEL_DLL_IMPORT __declspec(dllimport)
    #define KSIMD_KERNEL_DLL_EXPORT __declspec(dllexport)
#elif KSIMD_KERNEL_COMPILER_GCC || KSIMD_KERNEL_COMPILER_CLANG
    #define KSIMD_KERNEL_DLL_IMPORT __attribute__((visibility("default")))
    #define KSIMD_KERNEL_DLL_EXPORT __attribute__((visibility("default")))
#endif


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


/* macro for unit test */
/* #define KSIMD_KERNEL_IS_TESTING */

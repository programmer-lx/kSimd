#pragma once

#include "../platform.hpp"

KSIMD_NAMESPACE_BEGIN

// --------------------------------- KSIMD_DYN_INSTRUCTION str ---------------------------------
#define KSIMD_DYN_INSTRUCTION_SCALAR            Scalar
#define KSIMD_DYN_INSTRUCTION_SSE               SSE
#define KSIMD_DYN_INSTRUCTION_SSE2              SSE2
#define KSIMD_DYN_INSTRUCTION_SSE3              SSE3
#define KSIMD_DYN_INSTRUCTION_SSSE3             SSSE3
#define KSIMD_DYN_INSTRUCTION_SSE4_1            SSE4_1
#define KSIMD_DYN_INSTRUCTION_SSE4_2            SSE4_2
#define KSIMD_DYN_INSTRUCTION_AVX               AVX
#define KSIMD_DYN_INSTRUCTION_AVX2              AVX2
#define KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C    AVX2_FMA3_F16C

// instruction充当命名空间
#define KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, instruction) \
    &instruction::func_name,

#define KSIMD_DETAIL_ONE_EMPTY_FUNC

// ---------------------------------------------- Function table ----------------------------------------------
// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SCALAR)
#else
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// SSE
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
    #define KSIMD_DETAIL_SSE_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SSE)
#else
    #define KSIMD_DETAIL_SSE_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// SSE2
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE2)
    #define KSIMD_DETAIL_SSE2_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SSE2)
#else
    #define KSIMD_DETAIL_SSE2_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// SSE3
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE3)
    #define KSIMD_DETAIL_SSE3_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SSE3)
#else
    #define KSIMD_DETAIL_SSE3_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// SSE4.1
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE4_1)
    #define KSIMD_DETAIL_SSE4_1_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SSE4_1)
#else
    #define KSIMD_DETAIL_SSE4_1_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// AVX
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX)
    #define KSIMD_DETAIL_AVX_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_AVX)
#else
    #define KSIMD_DETAIL_AVX_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// AVX2
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2)
    #define KSIMD_DETAIL_AVX2_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_AVX2)
#else
    #define KSIMD_DETAIL_AVX2_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// AVX_FMA3_F16C
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2) && defined(KSIMD_INSTRUCTION_FEATURE_FMA3) && defined(KSIMD_INSTRUCTION_FEATURE_F16C)
    #define KSIMD_DETAIL_AVX2_FMA3_F16C_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C)
#else
    #define KSIMD_DETAIL_AVX2_FMA3_F16C_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// function table
#define KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name) \
    /* ------------------------------------- scalar ------------------------------------- */ \
    KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name) \
    /* ------------------------------------- x86 ------------------------------------- */ \
    KSIMD_DETAIL_SSE_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_SSE2_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_SSE3_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_SSE4_1_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_AVX_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_AVX2_FUNC_IMPL(func_name) \
    KSIMD_DETAIL_AVX2_FMA3_F16C_FUNC_IMPL(func_name)

#if !defined(KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY)
    #error "have not defined DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY to cache the simd function pointers"
#endif


// -------------------------- dispatch function ---------------------------
namespace detail
{
    // 这个枚举的值就是函数指针表的索引
    enum class SimdInstructionIndex : int
    {
        Invalid = -1,

    #if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
        Scalar,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
        SSE,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_SSE2)
        SSE2,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_SSE3)
        SSE3,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_SSE4_1)
        SSE4_1,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_AVX)
        AVX,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2)
        AVX2,
    #endif

    #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2) && defined(KSIMD_INSTRUCTION_FEATURE_FMA3) && defined(KSIMD_INSTRUCTION_FEATURE_F16C)
        AVX2_FMA3_F16C,
    #endif

        Num
    };
    static_assert(static_cast<int>(SimdInstructionIndex::Num) > 0);
}

// 测试时直接返回索引即可，正式版本才使用运行时CPUID判断
int dyn_func_index() noexcept;

#define KSIMD_DETAIL_PFN_TABLE_NS K_S_I_M_D__PFN_TABLE
#define KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name) KSIMD_DETAIL_PFN_TABLE_NS::func_name

#define KSIMD_DYN_DISPATCH_FUNC(func_name) \
    /* 构建静态数组，存储函数指针 (使用命名空间包裹，限定只能在类外使用)，不能添加 static, inline 声明，强制整个程序只能有一份函数表 */ \
    namespace KSIMD_DETAIL_PFN_TABLE_NS { \
        decltype(&KSIMD_DYN_INSTRUCTION::func_name) func_name[] = { \
            KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name) \
        }; \
    }

#define KSIMD_DYN_FUNC_POINTER(func_name) \
    KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)[KSIMD_NAMESPACE_NAME::dyn_func_index()]


#define KSIMD_DYN_CALL(func_name) (KSIMD_DYN_FUNC_POINTER(func_name))



// --------------------------------- FUNC_ATTR字符串描述 ---------------------------------
// 将会在 dispatch_this_file.hpp 文件被多次重定义
#define KSIMD_DYN_FUNC_ATTR "you should include <tSimd/dispatch_this_file.hpp>"

KSIMD_NAMESPACE_END

// utils : 工具库
#include "utils.inl"

// func attributes : 函数的 __attribute__((...)), __declspec(...)
#include "func_attr.inl"

// InstructionTraits : 指令集信息 (lanes, alignment, ...)
#include "SimdTraits.inl"

// SimdOp<SimdInstruction, ScalarType> : 用于SIMD数据操作
#include "SimdOp.inl"

// TypeOp<SimdInstruction> : 用于SIMD类型操作，比如 bitcast
#include "TypeOp.inl"

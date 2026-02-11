#pragma once

// 该头文件编写函数分发表相关逻辑

#include "base.hpp"

// --------------------------------- DISPATCH_LEVEL ---------------------------------
// 将会在 dispatch_this_file.hpp 文件被多次重定义
#define KSIMD_DYN_DISPATCH_LEVEL "we should include our file after include <kSimd/core/dispatch_this_file.hpp>"

// 该宏用于判断分发到了哪一层指令集
// KSIMD_DYN_DISPATCH_LEVEL values
#define KSIMD_DYN_DISPATCH_LEVEL_SCALAR         1

#define KSIMD_DYN_DISPATCH_LEVEL_AVX_START      2
#define KSIMD_DYN_DISPATCH_LEVEL_AVX2_MAX       3
#define KSIMD_DYN_DISPATCH_LEVEL_AVX_END        4


// --------------------------------- FUNC_ATTR ---------------------------------
// 将会在 dispatch_this_file.hpp 文件被多次重定义
#define KSIMD_DYN_FUNC_ATTR "we should include our file after include <kSimd/core/dispatch_this_file.hpp>"

/*
    see https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html#x86-Function-Attributes
    for more intrinsics attributes information.
*/

// scalar
#define KSIMD_DYN_FUNC_ATTR_SCALAR

// avx2+fma3+f16c
#define KSIMD_DYN_FUNC_ATTR_AVX2_MAX KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2,fma,f16c")


// --------------------------------- KSIMD_DYN_INSTRUCTION ---------------------------------
// 将会在 dispatch_this_file.hpp 文件被多次重定义
#define KSIMD_DYN_INSTRUCTION "we should include our file after include <kSimd/core/dispatch_this_file.hpp>"

#define KSIMD_DYN_INSTRUCTION_SCALAR   SCALAR
#define KSIMD_DYN_INSTRUCTION_AVX2_MAX AVX2_MAX

// avx2 max fallback
#if KSIMD_INSTRUCTION_FEATURE_AVX2_MAX == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #undef KSIMD_DYN_INSTRUCTION_FALLBACK
    #define KSIMD_DYN_INSTRUCTION_FALLBACK KSIMD_DYN_INSTRUCTION_AVX2_MAX
#endif

// scalar fallback
#if KSIMD_INSTRUCTION_FEATURE_SCALAR == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #undef KSIMD_DYN_INSTRUCTION_FALLBACK
    #define KSIMD_DYN_INSTRUCTION_FALLBACK KSIMD_DYN_INSTRUCTION_SCALAR
#endif

// check fallback
#if !defined(KSIMD_DYN_INSTRUCTION_FALLBACK)
    #error "We must define a fallback instruction name."
#endif


// instruction充当命名空间
#define KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, instruction) &instruction::func_name,

#define KSIMD_DETAIL_ONE_EMPTY_FUNC

// ---------------------------------------------- Function table ----------------------------------------------
// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SCALAR)
#else
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// AVX_FMA3_F16C
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
    #define KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(func_name) \
        KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_AVX2_MAX)
#else
    #define KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(func_name) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// function table
#define KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name) \
    /* ------------------------------------- avx family ------------------------------------- */ \
    KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(func_name) \
    /* ------------------------------------- scalar ------------------------------------- */ \
    KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name)

#if !defined(KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY)
    #error "have not defined DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY to cache the simd function pointers"
#endif

namespace ksimd
{
    // -------------------------- dispatch function ---------------------------
    namespace detail
    {
        // 这个枚举的值就是函数指针表的索引
        // 越现代的指令集，排得越靠前，索引越小
        enum class SimdInstructionIndex : int
        {
            Invalid = -1,

        #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
            KSIMD_DYN_INSTRUCTION_AVX2_MAX,
        #endif

        #if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
            KSIMD_DYN_INSTRUCTION_SCALAR,
        #endif

            Num
        };
        static_assert(static_cast<int>(SimdInstructionIndex::Num) > 0);
    }

    // 测试时直接返回索引即可，正式版本才使用运行时CPUID判断
    KSIMD_HEADER_GLOBAL int dyn_func_index() noexcept
    {
        static int i = []()
        {
            const CpuSupportInfo& supports = get_cpu_support_info();

            // 从最高级的指令往下判断
        #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
            if (supports.AVX2 && supports.FMA3 && supports.F16C)
            {
                return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX2_MAX);
            }
        #endif

            // 返回实际的 fallback index 即可，某些平台，标量可能不是 fallback
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_FALLBACK);
        }();

        return i;
    }
}

#define KSIMD_DETAIL_PFN_TABLE_NS K_S_I_M_D__PFN_TABLE
#define KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name) KSIMD_DETAIL_PFN_TABLE_NS::func_name

#define KSIMD_DYN_DISPATCH_FUNC(func_name) \
    /* 构建静态数组，存储函数指针 (使用命名空间包裹，限定只能在类外使用)，不能添加 static, inline 声明，强制整个程序只能有一份函数表 */ \
    namespace KSIMD_DETAIL_PFN_TABLE_NS { \
        const decltype(&KSIMD_DYN_INSTRUCTION::func_name) func_name[] = { \
            KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name) \
        }; \
    }

#define KSIMD_DYN_FUNC_POINTER(func_name) \
    KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)[ksimd::dyn_func_index()]
#define KSIMD_DYN_CALL(func_name) (KSIMD_DYN_FUNC_POINTER(func_name))

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

// popcnt
#define KSIMD_DYN_FUNC_ATTR_POPCNT KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("popcnt")

// sse4.2
#define KSIMD_DYN_FUNC_ATTR_SSE42 KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("sse4.2")

// avx2
#define KSIMD_DYN_FUNC_ATTR_AVX2 KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2")

// avx2+fma3+f16c
#define KSIMD_DYN_FUNC_ATTR_AVX2_MAX KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2,fma,f16c")


// --------------------------------- KSIMD_DYN_INSTRUCTION ---------------------------------
// 将会在 dispatch_this_file.hpp 文件被多次重定义
#define KSIMD_DYN_INSTRUCTION "we should include our file after include <kSimd/core/dispatch_this_file.hpp>"

#define KSIMD_DYN_INSTRUCTION_SCALAR   KSIMD_SCALAR
#define KSIMD_DYN_INSTRUCTION_AVX2_MAX KSIMD_AVX2_MAX

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


// instruction充当命名空间, __VA_ARGS__ 是模板实参
#define KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, instruction, ...) &instruction::func_name __VA_ARGS__,

#define KSIMD_DETAIL_ONE_EMPTY_FUNC

// ---------------------------------------------- Function table ----------------------------------------------
// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name, ...) \
        KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_SCALAR, __VA_ARGS__)
#else
    #define KSIMD_DETAIL_SCALAR_FUNC_IMPL(...) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// AVX_FMA3_F16C
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
    #define KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(func_name, ...) \
        KSIMD_DETAIL_ONE_FUNC_IMPL(func_name, KSIMD_DYN_INSTRUCTION_AVX2_MAX, __VA_ARGS__)
#else
    #define KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(...) KSIMD_DETAIL_ONE_EMPTY_FUNC
#endif

// function table
#define KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name, ...) \
    /* ------------------------------------- avx family ------------------------------------- */ \
    KSIMD_DETAIL_AVX2_MAX_FUNC_IMPL(func_name, __VA_ARGS__) \
    /* ------------------------------------- scalar ------------------------------------- */ \
    KSIMD_DETAIL_SCALAR_FUNC_IMPL(func_name, __VA_ARGS__)

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

        // 必须使用 内部链接 使每份CPP文件拥有一个单独的函数，这样，就能通过宏定义，来单独的控制每份CPP文件需要分发哪些指令集，不分发哪些指令集
        [[maybe_unused]] static int dyn_func_index() noexcept
        {
            static int i = []()
            {
                [[maybe_unused]] const ksimd::CpuSupportInfo& supports = ksimd::get_cpu_support_info();

                // 从最高级的指令往下判断
                #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
                if (supports.AVX2 && supports.FMA3 && supports.F16C)
                {
                    return ksimd::detail::underlying(ksimd::detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX2_MAX);
                }
                #endif

                // 返回实际的 fallback index 即可，某些平台，标量可能不是 fallback
                return ksimd::detail::underlying(ksimd::detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_FALLBACK);
            }();

            return i;
        }
    }
}

#define KSIMD_DETAIL_PFN_TABLE_PREFIX KSIMD_PFN_TABLE_
#define KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name) KSIMD_CONCAT(KSIMD_DETAIL_PFN_TABLE_PREFIX, func_name)

// __VA_ARGS__ 是模板的完整实参，比如 <T1, T2>，不要漏掉尖括号
#define KSIMD_DYN_DISPATCH_FUNC(func_name, ...) \
    /* 构建静态数组，存储函数指针，不能添加 static, inline 声明，强制整个程序只能有一份函数表 */ \
    const decltype(&KSIMD_DYN_INSTRUCTION::func_name __VA_ARGS__) KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)[] = { \
        KSIMD_DETAIL_DYN_DISPATCH_FUNC_POINTER_STATIC_ARRAY(func_name, __VA_ARGS__) \
    }

// __VA_ARGS__ 是模板完整实参
#define KSIMD_DYN_FUNC_POINTER(func_name, ...) \
    KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name) __VA_ARGS__ [ksimd::detail::dyn_func_index()]

// __VA_ARGS__ 是模板完整实参
#define KSIMD_DYN_CALL(func_name, ...) (KSIMD_DYN_FUNC_POINTER(func_name, __VA_ARGS__))

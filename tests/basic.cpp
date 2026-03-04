#include "test.hpp"

#include <string>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "basic.cpp" // this file
#include <kSimd/core/dispatch_this_file.hpp>

#include <kSimd/core/dispatch_core.hpp>

#pragma message("dispatch intrinsic: \"" KSIMD_STR("" KSIMD_DYN_FUNC_ATTR) "\"")


#if KSIMD_COMPILER_MSVC
    #pragma message("compiler: msvc")
#endif

#if KSIMD_COMPILER_GCC
    #pragma message("compiler: GCC")
#endif

#if KSIMD_COMPILER_CLANG
    #pragma message("compiler: clang")
#endif

#if KSIMD_COMPILER_CLANG_CL
    #pragma message("compiler: clang-cl")
#endif

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR void kernel_dyn_impl(int index) noexcept
    {
        #if KSIMD_ARCH_X86_ANY

        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);

        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 3) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V2) && index == 2) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V3) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V4) && index == 0);

        EXPECT_TRUE(result);

        #elif KSIMD_ARCH_ARM_64

        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        
        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_NEON) && index == 0);

        EXPECT_TRUE(result);

        #else
        #error unknown test arch
        #endif
    }
}


#if KSIMD_ONCE

// export impl function
KSIMD_DYN_DISPATCH_FUNC(kernel_dyn_impl);

TEST(dyn_dispatch, pfn_table)
{
#if KSIMD_ARCH_X86_ANY

    // 0: avx512 v4
    // 1: avx v3
    // 2: sse v2
    // 3: scalar
    EXPECT_EQ(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)), 4);

#elif KSIMD_ARCH_ARM_ANY

    // 0: neon
    // 1: scalar
    EXPECT_EQ(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)), 2);

#endif

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)[i]((int)i);
    }
}

template<typename T>
void test_scalar_op_constants()
{
    namespace ns = ksimd::KSIMD_DYN_INSTRUCTION_SCALAR;
    EXPECT_TRUE(ns::lanes(ns::FullTag<T>{}) == (16 / sizeof(T)));
}

TEST(dyn_dispatch, constants)
{
    test_scalar_op_constants<float>();
    test_scalar_op_constants<double>();
    test_scalar_op_constants<int8_t>();
    test_scalar_op_constants<uint8_t>();
    test_scalar_op_constants<int16_t>();
    test_scalar_op_constants<uint16_t>();
    test_scalar_op_constants<int32_t>();
    test_scalar_op_constants<uint32_t>();
    test_scalar_op_constants<int64_t>();
    test_scalar_op_constants<uint64_t>();
}

TEST(std_float_types, basic)
{
    #if KSIMD_SUPPORT_STD_FLOAT16
    static_assert(!std::is_same_v<std::float16_t, uint16_t>);
    static_assert(sizeof(std::float16_t) == sizeof(uint16_t));
    #endif

    #if KSIMD_SUPPORT_STD_FLOAT32
    static_assert(!std::is_same_v<std::float32_t, float>);
    static_assert(sizeof(std::float32_t) == sizeof(float));
    #endif

    #if KSIMD_SUPPORT_STD_FLOAT64
    static_assert(!std::is_same_v<std::float64_t, double>);
    static_assert(sizeof(std::float64_t) == sizeof(double));
    #endif

    SUCCEED();
}

TEST(scalar_funcs, basic)
{
    EXPECT_FALSE(ksimd::is_NaN(std::numeric_limits<float>::infinity()));
    EXPECT_FALSE(ksimd::is_NaN(std::numeric_limits<double>::infinity()));

    EXPECT_FALSE(ksimd::is_finite(std::numeric_limits<float>::infinity()));
    EXPECT_FALSE(ksimd::is_finite(std::numeric_limits<double>::infinity()));
    EXPECT_TRUE(ksimd::is_finite(123.f));
    EXPECT_TRUE(ksimd::is_finite(123.0));

    EXPECT_FALSE(ksimd::is_inf(123.0f));
    EXPECT_FALSE(ksimd::is_inf(123.0));
    EXPECT_TRUE(ksimd::is_inf(std::numeric_limits<float>::infinity()));
    EXPECT_TRUE(ksimd::is_inf(std::numeric_limits<double>::infinity()));
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
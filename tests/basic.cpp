#include "test.hpp"

#include <string>
#include <stdfloat>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "basic.cpp" // this file
#include <kSimd/core/dispatch_this_file.hpp>

#include <kSimd/core/dispatch_core.hpp>

#pragma message("dispatch intrinsic: \"" KSIMD_STR("" KSIMD_DYN_FUNC_ATTR) "\"")

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR void kernel_dyn_impl(const float*, const size_t, float*) noexcept
    {
    }
}


#if KSIMD_ONCE

// export impl function
KSIMD_DYN_DISPATCH_FUNC(kernel_dyn_impl);

TEST(dyn_dispatch, pfn_table_size)
{
    EXPECT_EQ(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)), (size_t)ksimd::detail::SimdInstructionIndex::Num);
}

template<typename T>
void test_scalar_op_constants()
{
    namespace ns = ksimd::KSIMD_DYN_INSTRUCTION_SCALAR;
    static_assert(ns::Lanes<T> == 1);
    static_assert(ns::Alignment<T> == alignof(T));
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

    SUCCEED();
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

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
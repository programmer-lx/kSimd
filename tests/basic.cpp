#include "test.hpp"

#include <string>

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
    test_scalar_op_constants<ksimd::float32>();
    test_scalar_op_constants<ksimd::float64>();
    test_scalar_op_constants<ksimd::int8>();
    test_scalar_op_constants<ksimd::uint8>();
    test_scalar_op_constants<ksimd::int16>();
    test_scalar_op_constants<ksimd::uint16>();
    test_scalar_op_constants<ksimd::int32>();
    test_scalar_op_constants<ksimd::uint32>();
    test_scalar_op_constants<ksimd::int64>();
    test_scalar_op_constants<ksimd::uint64>();

    SUCCEED();
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
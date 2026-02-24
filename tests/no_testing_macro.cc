
#include <gtest/gtest.h>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "no_testing_macro.cc" // this file
#include <kSimd/core/dispatch_this_file.hpp>

#include <kSimd/core/dispatch_core.hpp>

#ifdef KSIMD_IS_TESTING
    #error no KSIMD_IS_TESTING
#endif

namespace KSIMD_DYN_INSTRUCTION
{
    void test_table_size()
    {}
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(test_table_size);
TEST(table_size, basic)
{
#if KSIMD_ARCH_X86_ANY

    // x86: AVX2_MAX + SCALAR == 2
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 2);

#elif KSIMD_ARCH_ARM_64

    // arm64: NEON == 1
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 1);

#elif KSIMD_ARCH_ARM_32

    // arm32: NEON + SCALAR == 2
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 2);

#else
    #error unknown arch
#endif

    SUCCEED();
}
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif

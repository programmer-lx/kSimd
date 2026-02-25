
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
    void test_table_size(size_t index)
    {
        #ifdef KSIMD_ARCH_X86_ANY

        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);

        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 2) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SSE4_1) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_AVX2_FMA3) && index == 0);

        EXPECT_TRUE(result);

        #elif KSIMD_ARCH_ARM_64

        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);

        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_NEON) && index == 0);

        EXPECT_TRUE(result);

        #elif KSIMD_ARCH_ARM_32

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
KSIMD_DYN_DISPATCH_FUNC(test_table_size);
TEST(table_size, basic)
{
#if KSIMD_ARCH_X86_ANY

    // x86: AVX2_FMA3 + SSE4_1 + SCALAR == 3
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 3);

#elif KSIMD_ARCH_ARM_64

    // arm64: NEON == 1
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 1);

#elif KSIMD_ARCH_ARM_32

    // arm32: NEON + SCALAR == 2
    static_assert(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)) == 2);

#else
    #error unknown arch
#endif

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(test_table_size)[i](i);
    }

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

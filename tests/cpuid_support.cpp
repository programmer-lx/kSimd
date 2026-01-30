#include <kSimd/simd_op.hpp>

#include "test.hpp"

TEST(cpuid, support)
{
    const auto& result = ksimd::get_cpu_support_info();

    EXPECT_TRUE(result.FXSR == true);

    EXPECT_TRUE(result.SSE == true);
    EXPECT_TRUE(result.SSE2 == true);
    EXPECT_TRUE(result.SSE3 == true);
    EXPECT_TRUE(result.SSSE3 == true);
    EXPECT_TRUE(result.SSE4_1 == true);
    EXPECT_TRUE(result.SSE4_2 == true);

    EXPECT_TRUE(result.OS_XSAVE == true);
    EXPECT_TRUE(result.XSAVE == true);
    EXPECT_TRUE(result.AVX == true);
    EXPECT_TRUE(result.F16C == true);
    EXPECT_TRUE(result.FMA3 == true);
    EXPECT_TRUE(result.AVX2 == true);

    // EXPECT_TRUE(result.AVX512_F == true); // not support
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
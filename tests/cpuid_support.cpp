#include <kSimd/core/impl/base.hpp>
#include <kSimd/core/impl/dispatch.hpp>

#include "test.hpp"

#if defined(KSIMD_ARCH_X86_ANY)
    #include <immintrin.h>
#endif

#include "kSimd/IDE/IDE_hint.hpp"

TEST(cpuid, support)
{
    [[maybe_unused]] const auto& result = ksimd::get_cpu_support_info();

#if defined(KSIMD_ARCH_X86_ANY)

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

    // EXPECT_TRUE(result.AVX512_F == true);

#elif defined(KSIMD_ARCH_ARM_ANY)

    EXPECT_TRUE(result.NEON == true);
    EXPECT_TRUE(result.SVE == true);

#else
    #error unknown arch
#endif
}

#if defined(KSIMD_ARCH_X86_ANY)
// TEST(avx512_test, test)
// {
//     [[maybe_unused]] bool result = []() KSIMD_DYN_FUNC_ATTR_AVX512F
//     {
//         [[maybe_unused]] auto avx512f_var = _mm512_set1_ps(1.0f);
//         return true;
//     }();
// }
#endif

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
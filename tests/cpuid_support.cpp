#include <kSimd/core/impl/base.hpp>
#include <kSimd/core/impl/dispatch.hpp>

#include "test.hpp"

#if defined(KSIMD_ARCH_X86_ANY)
    #include <immintrin.h>
#endif

#if defined(KSIMD_ARCH_ARM_ANY)
    #include <arm_sve.h>
#endif

#include "kSimd/IDE/IDE_hint.hpp"

TEST(cpuid, support)
{
    [[maybe_unused]] const auto& result = ksimd::get_cpu_support_info();

#if KSIMD_ARCH_X86_ANY

    EXPECT_TRUE(result.POPCNT == true);
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

    EXPECT_TRUE(result.AVX512_F == true); // 可在SDE环境下模拟AVX512F指令

#elif KSIMD_ARCH_ARM_ANY

    EXPECT_TRUE(reuslt.ARM_CRC32 == true);
    EXPECT_TRUE(result.NEON == true);
    EXPECT_TRUE(result.SVE == true);

#else
    #error unknown arch
#endif
}

#if KSIMD_ARCH_X86_ANY
TEST(avx512_test, test)
{
    [[maybe_unused]] bool result = []() KSIMD_DYN_FUNC_ATTR_AVX512F
    {
        [[maybe_unused]] auto avx512f_var = _mm512_set1_ps(1.0f);
        return true;
    }();
}
#endif

#if KSIMD_ARCH_ARM_ANY
TEST(sve_test, sve_vector_size)
{
    [[maybe_unused]] float a = []() KSIMD_DYN_FUNC_ATTR_SVE
    {
        svbool_t pg = svptrue_b32();

        svfloat32_t v1 = svdup_f32(1.0f);
        svfloat32_t v2 = svdup_f32(2.0f);

        svfloat32_t res = svadd_f32_z(pg, v1, v2);

        return svlasta_f32(pg, res);
    }();

    #if KSIMD_TEST_SVE_BITS == 128
    #pragma message("SVE bits = 128")
    [[maybe_unused]] bool b = []() KSIMD_DYN_FUNC_ATTR_SVE
    {
        EXPECT_EQ(svcntb(), 16); // SVE-128
        return true;
    }();
    #endif

    #if KSIMD_TEST_SVE_BITS == 256
    #pragma message("SVE bits = 256")
    [[maybe_unused]] bool b = []() KSIMD_DYN_FUNC_ATTR_SVE
    {
        EXPECT_EQ(svcntb(), 32); // SVE-256
        return true;
    }();
    #endif

    #if KSIMD_TEST_SVE_BITS == 512
    #pragma message("SVE bits = 512")
    [[maybe_unused]] bool b = []() KSIMD_DYN_FUNC_ATTR_SVE
    {
        EXPECT_EQ(svcntb(), 64); // SVE-512
        return true;
    }();
    #endif
}
#endif

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
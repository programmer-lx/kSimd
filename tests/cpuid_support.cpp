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

    EXPECT_TRUE(result.popcnt == true);
    EXPECT_TRUE(result.aes_ni == true);
    EXPECT_TRUE(result.sha == true);

    EXPECT_TRUE(result.fxsr == true);

    // sse family
    EXPECT_TRUE(result.sse == true);
    EXPECT_TRUE(result.sse2 == true);
    EXPECT_TRUE(result.sse3 == true);
    EXPECT_TRUE(result.ssse3 == true);
    EXPECT_TRUE(result.sse4_1 == true);
    EXPECT_TRUE(result.sse4_2 == true);

    EXPECT_TRUE(result.os_xsave == true);
    EXPECT_TRUE(result.xsave == true);

    // avx family
    EXPECT_TRUE(result.avx == true);
    EXPECT_TRUE(result.f16c == true);
    EXPECT_TRUE(result.fma3 == true);
    EXPECT_TRUE(result.avx2 == true);

    EXPECT_TRUE(result.avx_vnni == true);
    EXPECT_TRUE(result.avx_vnni_int8 == true);
    EXPECT_TRUE(result.avx_ne_convert == true);
    EXPECT_TRUE(result.avx_ifma == true);
    EXPECT_TRUE(result.avx_vnni_int16 == true);
    EXPECT_TRUE(result.sha512 == true);
    EXPECT_TRUE(result.sm3 == true);
    EXPECT_TRUE(result.sm4 == true);

    // avx512 family
    EXPECT_TRUE(result.avx512_f == true); // 可在SDE环境下模拟AVX512F指令
    EXPECT_TRUE(result.avx512_bw == true);
    EXPECT_TRUE(result.avx512_cd == true);
    EXPECT_TRUE(result.avx512_dq == true);
    EXPECT_TRUE(result.avx512_ifma == true);
    EXPECT_TRUE(result.avx512_vl == true);
    EXPECT_TRUE(result.avx512_vpopcntdq == true);
    EXPECT_TRUE(result.avx512_bf16 == true);
    EXPECT_TRUE(result.avx512_bitalg == true);
    EXPECT_TRUE(result.avx512_vbmi == true);
    EXPECT_TRUE(result.avx512_vbmi2 == true);
    EXPECT_TRUE(result.avx512_vnni == true);
    // EXPECT_TRUE(result.avx512_vp2intersect == true); // SDE future CPU 不支持
    EXPECT_TRUE(result.avx512_fp16 == true);

#elif KSIMD_ARCH_ARM_ANY

    EXPECT_TRUE(result.arm_crc32 == true);
    EXPECT_TRUE(result.neon == true);
    EXPECT_TRUE(result.sve == true);

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
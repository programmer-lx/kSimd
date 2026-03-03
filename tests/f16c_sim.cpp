#include "test.hpp"
#include <fp16.h>

#include "kSimd/core/impl/dispatch.hpp"

#if KSIMD_ARCH_X86_ANY
#undef KSIMD_DYN_INSTRUCTION
#define KSIMD_DYN_INSTRUCTION KSIMD_DYN_INSTRUCTION_X86_V2

#undef KSIMD_DYN_DISPATCH_LEVEL
#define KSIMD_DYN_DISPATCH_LEVEL KSIMD_DYN_DISPATCH_LEVEL_X86_V2

#undef KSIMD_DYN_FUNC_ATTR
#define KSIMD_DYN_FUNC_ATTR KSIMD_DYN_FUNC_ATTR_X86_V2

#include "kSimd/core/impl/ops/x86_vec128.hpp"

namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;

TEST(f16_to_f32, x86)
{
    []() KSIMD_DYN_FUNC_ATTR_X86_V2
    {
        {
            alignas(16) uint16_t f16[] = {
                fp16_ieee_from_fp32_value(0.0f),
                fp16_ieee_from_fp32_value(-0.0f),
                fp16_ieee_from_fp32_value(12.123f),
                fp16_ieee_from_fp32_value(-45.56f)
            };

            __m128i f16_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f16));
            __m128 f32_v = ns::detail::mm_f16_to_f32(f16_v);

            alignas(16) float f32[4];
            _mm_store_ps(f32, f32_v);

            EXPECT_EQ(f32[0], 0.0f);
            EXPECT_EQ(f32[1], -0.0f);
            EXPECT_TRUE(std::signbit(f32[1]));
            EXPECT_NEAR(f32[2], 12.123f, 0.01f);
            EXPECT_NEAR(f32[3], -45.56f, 0.01f);
        }

        // nan inf
        {
            alignas(16) uint16_t f16[] = {
                fp16_ieee_from_fp32_value(0.0f),
                fp16_ieee_from_fp32_value(-inf<float>),
                fp16_ieee_from_fp32_value(inf<float>),
                fp16_ieee_from_fp32_value(qNaN<float>)
            };

            __m128i f16_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f16));
            __m128 f32_v = ns::detail::mm_f16_to_f32(f16_v);

            alignas(16) float f32[4];
            _mm_store_ps(f32, f32_v);

            EXPECT_EQ(f32[0], 0.0f);
            EXPECT_EQ(f32[1], -inf<float>);
            EXPECT_TRUE(std::signbit(f32[1]));

            EXPECT_EQ(f32[2], inf<float>);
            EXPECT_TRUE(!std::signbit(f32[2]));

            EXPECT_TRUE(std::isnan(f32[3]));
        }

        // 极值与非规格化数 (Denormal & Max Normal)
        {
            alignas(16) uint16_t f16[] = {
                0x0001, // 最小的非规格化正数 (Min Denormal): 2^-24 ≈ 5.96e-8
                0x03FF, // 最大的非规格化正数 (Max Denormal): ≈ 6.10e-5
                0x7BFF, // 最大的规格化正数 (Max Normal): 65504.0f
                0xFBFF  // 最小的规格化负数 (Max Negative Normal): -65504.0f
            };

            __m128i f16_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f16));
            __m128 f32_v = ns::detail::mm_f16_to_f32(f16_v);

            alignas(16) float f32[4];
            _mm_store_ps(f32, f32_v);

            // 验证非规格化数 (Denormals)
            EXPECT_NEAR(f32[0], 5.96046e-8f, 1e-12f);
            EXPECT_NEAR(f32[1], 6.09756e-5f, 1e-7f);

            // 验证大数健壮性 (Max Normal)
            // 如果算法溢出或符号位错误，这里会变成 Inf 或负数
            EXPECT_EQ(f32[2], 65504.0f);
            EXPECT_EQ(f32[3], -65504.0f);
        }

        // 精度与指数边界
        {
            alignas(16) uint16_t f16[] = {
                0x3C00, // 1.0f (指数偏移的正中心)
                0x3C01, // 1.0009765625f (最小精度步进)
                0x0400, // 最小规格化正数 (Min Normal): 2^-14 ≈ 6.1035e-5
                0x8400  // 最小规格化负数: -6.1035e-5
            };

            __m128i f16_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f16));
            __m128 f32_v = ns::detail::mm_f16_to_f32(f16_v);

            alignas(16) float f32[4];
            _mm_store_ps(f32, f32_v);

            EXPECT_EQ(f32[0], 1.0f);
            EXPECT_NEAR(f32[1], 1.0009765625f, 0.0001f);

            // 验证规格化边界
            EXPECT_NEAR(f32[2], 6.1035e-5f, 1e-9f);
            EXPECT_NEAR(f32[3], -6.1035e-5f, 1e-9f);
        }

        // 复杂的 NaN 与符号位
        {
            alignas(16) uint16_t f16[] = {
                0xFE00, // 带有符号位的 NaN (-NaN)
                0x7E01, // Signaling NaN (SNaN)
                0xFC00, // -Infinity (重复确认)
                0x0000  // +0.0 (重复确认)
            };

            __m128i f16_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(f16));
            __m128 f32_v = ns::detail::mm_f16_to_f32(f16_v);

            alignas(16) float f32[4];
            _mm_store_ps(f32, f32_v);

            // 验证 -NaN
            EXPECT_TRUE(std::isnan(f32[0]));
            EXPECT_TRUE(std::signbit(f32[0])); // 验证符号位是否保留

            // 验证 SNaN 是否依然是 NaN
            EXPECT_TRUE(std::isnan(f32[1]));

            EXPECT_EQ(f32[2], -inf<float>);
            EXPECT_EQ(f32[3], 0.0f);
        }
    }();
}
#endif

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
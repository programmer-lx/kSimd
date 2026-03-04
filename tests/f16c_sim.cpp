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

TEST(f32_to_f16, x86)
{
    []() KSIMD_DYN_FUNC_ATTR_X86_V2
    {
        // 基础数值与精度截断 (Normal values & Rounding)
        {
            alignas(16) float f32[] = {
                0.0f,
                -0.0f,
                12.123f,  // 会被舍入到最接近的 FP16 (12.125)
                -45.56f   // 会被舍入到 -45.5625
            };

            __m128 f32_v = _mm_load_ps(f32);
            // 假设你的 store 逻辑最终返回一个包含 4 个 uint16_t 的 __m128i 低 64 位
            __m128i f16_v = ns::detail::mm_f32_to_f16(f32_v);

            alignas(16) uint16_t f16[8]; // 只读取前 4 个
            _mm_storel_epi64(reinterpret_cast<__m128i*>(f16), f16_v);

            EXPECT_EQ(f16[0], 0x0000); // 0.0
            EXPECT_EQ(f16[1], 0x8000); // -0.0

            // 验证舍入后的位模式
            EXPECT_EQ(f16[2], fp16_ieee_from_fp32_value(12.123f));
            EXPECT_EQ(f16[3], fp16_ieee_from_fp32_value(-45.56f));
        }

        // NaN, Inf 与 溢出边界 (NaN, Inf & Overflow)
        {
            alignas(16) float f32[] = {
                inf<float>,
                -inf<float>,
                qNaN<float>,
                70000.0f  // 超过 65504，应该溢出到 FP16 的 Inf
            };

            __m128 f32_v = _mm_load_ps(f32);
            __m128i f16_v = ns::detail::mm_f32_to_f16(f32_v);

            alignas(16) uint16_t f16[4];
            _mm_storel_epi64(reinterpret_cast<__m128i*>(f16), f16_v);

            EXPECT_EQ(f16[0], 0x7C00); // +inf
            EXPECT_EQ(f16[1], 0xFC00); // -inf
            EXPECT_TRUE((f16[2] & 0x7C00) == 0x7C00 && (f16[2] & 0x03FF) != 0); // Is NaN
            EXPECT_EQ(f16[3], 0x7C00); // 70000.0 -> +inf
        }

        // 极小值与下溢 (Small values & Underflow)
        {
            alignas(16) float f32[] = {
                6.1035e-5f,  // FP16 最小规格化数 (0x0400)
                5.9604e-8f,  // FP16 最小非规格化数 (0x0001)
                1.0e-10f,    // 太小了，应该下溢到 0.0
                -1.0e-10f    // 应该下溢到 -0.0
            };

            __m128 f32_v = _mm_load_ps(f32);
            __m128i f16_v = ns::detail::mm_f32_to_f16(f32_v);

            alignas(16) uint16_t f16[4];
            _mm_storel_epi64(reinterpret_cast<__m128i*>(f16), f16_v);

            EXPECT_EQ(f16[0], 0x0400);
            EXPECT_EQ(f16[1], 0x0001);
            EXPECT_EQ(f16[2], 0x0000);
            EXPECT_EQ(f16[3], 0x8000);
        }

        // 符号位与最大规格化数
        {
            alignas(16) float f32[] = {
                65504.0f,    // FP16 Max Normal
                -65504.0f,   // FP16 Min Normal
                1.0f,
                -1.0f
            };

            __m128 f32_v = _mm_load_ps(f32);
            __m128i f16_v = ns::detail::mm_f32_to_f16(f32_v);

            alignas(16) uint16_t f16[4];
            _mm_storel_epi64(reinterpret_cast<__m128i*>(f16), f16_v);

            EXPECT_EQ(f16[0], 0x7BFF);
            EXPECT_EQ(f16[1], 0xFBFF);
            EXPECT_EQ(f16[2], 0x3C00);
            EXPECT_EQ(f16[3], 0xBC00);
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
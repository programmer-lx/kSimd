#include "../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/f16_load_store.cpp" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/base_op.hpp>

using namespace ksimd;

// ------------------------------------------ fp16_io_aligned ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void fp16_io_aligned() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(float32);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(op::BatchAlignment) float32 input_f32[Lanes];
        alignas(16) uint16_t mid_f16[Lanes];
        alignas(op::BatchAlignment) float32 output_f32[Lanes];

        // 构造测试数据
        for (size_t i = 0; i < Lanes; ++i) {
            if (i % 8 == 0)      input_f32[i] = 1.5f;          // Ties to even (Round to 2.0)
            else if (i % 8 == 1) input_f32[i] = 2.5f;          // Ties to even (Round to 2.0)
            else if (i % 8 == 2) input_f32[i] = 65504.0f;      // FP16 Max
            else                 input_f32[i] = static_cast<float>(i) * 0.33f;
        }

        // --- 1. Store 测试 (F32 -> F16) ---
        auto batch = op::load(input_f32);
        op::store_float16(reinterpret_cast<float16*>(mid_f16), batch);

        for (size_t i = 0; i < Lanes; ++i) {
            // 对标 Facebook 库的转换函数
            uint16_t expected = fp16_ieee_from_fp32_value(input_f32[i]);
            EXPECT_EQ(mid_f16[i], expected) << "Aligned Store Mismatch at index " << i;
        }

        // --- 2. Load 测试 (F16 -> F32) ---
        auto reloaded = op::load_float16(reinterpret_cast<const float16*>(mid_f16));
        op::store(output_f32, reloaded);

        for (size_t i = 0; i < Lanes; ++i) {
            // 验证 Load 后的浮点值是否与库函数还原的值完全一致
            float expected = fp16_ieee_to_fp32_value(mid_f16[i]);
            EXPECT_EQ(output_f32[i], expected) << "Aligned Load Mismatch at index " << i;
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(fp16_io_aligned)
#endif

// ------------------------------------------ fp16_io_unaligned ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void fp16_io_unaligned() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(float32);
        constexpr size_t Lanes = op::TotalLanes;

        // 构造非对齐内存环境
        alignas(op::BatchAlignment) float32 input_f32_raw[Lanes + 1];
        alignas(op::BatchAlignment) uint16_t mid_f16_raw[Lanes + 1];
        alignas(op::BatchAlignment) float32 output_f32_raw[Lanes + 1];

        // 偏移 1 个元素，使地址不再满足 16/32 字节对齐
        float32* u_input = input_f32_raw + 1;
        uint16_t* u_mid = mid_f16_raw + 1;
        float32* u_output = output_f32_raw + 1;

        for (size_t i = 0; i < Lanes; ++i) {
            u_input[i] = static_cast<float>(i) * -0.75f;
        }

        // --- 1. 使用 storeu_float16 ---
        auto batch = op::loadu(u_input);
        op::storeu_float16(reinterpret_cast<float16*>(u_mid), batch);

        for (size_t i = 0; i < Lanes; ++i) {
            uint16_t expected = fp16_ieee_from_fp32_value(u_input[i]);
            EXPECT_EQ(u_mid[i], expected) << "Unaligned Store Mismatch at index " << i;
        }

        // --- 2. 使用 loadu_float16 ---
        auto reloaded = op::loadu_float16(reinterpret_cast<const float16*>(u_mid));
        op::storeu(u_output, reloaded);

        for (size_t i = 0; i < Lanes; ++i) {
            float expected = fp16_ieee_to_fp32_value(u_mid[i]);
            EXPECT_EQ(u_output[i], expected) << "Unaligned Load Mismatch at index " << i;
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(fp16_io_unaligned)
#endif


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
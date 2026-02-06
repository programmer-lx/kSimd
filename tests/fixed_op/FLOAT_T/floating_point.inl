#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "fixed_op/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/fixed_op.hpp>

using namespace ksimd;


// ------------------------------------------ dot ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void dot() noexcept
    {
        using op = KSIMD_DYN_FIXED_OP_COUNT(FLOAT_T, LANES, COUNT);

        constexpr size_t TOTAL = op::TotalLanes;
        constexpr size_t STRIDE = op::RegLanes;

        alignas(op::BatchAlignment) FLOAT_T res[TOTAL]{};
        // 针对点积累加放大，使用更稳健的容差
        constexpr FLOAT_T eps = std::numeric_limits<FLOAT_T>::epsilon() * FLOAT_T(10);

        // 1. 准备动态数据
        // 注意：op::sequence(start) 通常生成 [start, start+1, start+2, ...]
        // 我们需要根据每个 Stride (4 lanes) 的实际数值来计算预期点积和
        auto va = op::sequence(1);
        auto vb = op::sequence(2);

        // --- Case 1: 测试 src_mask (X|Y) -> dst_mask (All) ---
        // 逻辑：每个寄存器单元内，取前两个通道点乘，结果广播到该单元所有通道
        op::store(res, op::template dot<op::X | op::Y, op::All>(va, vb));
        for (size_t r = 0; r < op::RegCount; ++r) {
            size_t base = r * STRIDE;
            // 动态计算该 Stride 的预期值: a0*b0 + a1*b1
            FLOAT_T a0 = FLOAT_T(base + 1), a1 = FLOAT_T(base + 2);
            FLOAT_T b0 = FLOAT_T(base + 2), b1 = FLOAT_T(base + 3);
            FLOAT_T expected_sum = a0 * b0 + a1 * b1;

            for (size_t l = 0; l < STRIDE; ++l) {
                EXPECT_NEAR(res[base + l], expected_sum, eps)
                    << "Error at Reg[" << r << "] Lane[" << l << "] (X|Y -> All)";
            }
        }

        // --- Case 2: 测试 dst_mask (仅存入 Z) ---
        // 逻辑：全通道参与计算，但结果只存在索引为 2 的通道
        op::store(res, op::template dot<op::All, op::Z>(va, vb));
        for (size_t r = 0; r < op::RegCount; ++r) {
            size_t base = r * STRIDE;
            // 计算全通道 (X,Y,Z,W) 的点积和
            FLOAT_T sum = 0;
            for (size_t l = 0; l < 4; ++l) {
                sum += FLOAT_T(base + l + 1) * FLOAT_T(base + l + 2);
            }

            for (size_t l = 0; l < STRIDE; ++l) {
                FLOAT_T expected = (l == 2) ? sum : FLOAT_T(0);
                EXPECT_NEAR(res[base + l], expected, eps)
                    << "Error at Reg[" << r << "] Lane[" << l << "] (All -> Z)";
            }
        }

        // --- Case 3: 边界测试 (负数与混合掩码 X|W) ---
        auto vc = op::set(FLOAT_T(-2.5));
        auto vd = op::set(FLOAT_T(4.0));
        // sum = (-2.5 * 4.0) * 2 (因为 src 只有 X, W) = -20.0
        op::store(res, op::template dot<op::X | op::W, op::X | op::W>(vc, vd));

        for (size_t i = 0; i < TOTAL; ++i) {
            size_t lane_in_reg = i % STRIDE;
            FLOAT_T expected = (lane_in_reg == 0 || lane_in_reg == 3) ? FLOAT_T(-20) : FLOAT_T(0);
            EXPECT_NEAR(res[i], expected, eps) << "Error at index " << i << " (Negative X|W)";
        }

        // --- Case 4: 零掩码测试 (None) ---
        op::store(res, op::template dot<op::None, op::All>(va, vb));
        for (size_t i = 0; i < TOTAL; ++i) {
            EXPECT_NEAR(res[i], FLOAT_T(0), eps);
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(dot)
#endif


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif

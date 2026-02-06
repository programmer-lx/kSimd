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

        constexpr size_t TOTAL = op::TotalLanes;    // 这里是 8
        // constexpr size_t STRIDE = op::RegLanes;     // SSE 为 4, AVX 为 8
        constexpr size_t DOT_UNIT = LANES;              // dot 指令的物理结算单元永远是 4

        alignas(op::BatchAlignment) FLOAT_T res[TOTAL]{};
        constexpr FLOAT_T eps = std::numeric_limits<FLOAT_T>::epsilon() * FLOAT_T(10);

        auto va = op::sequence(1); // [1, 2, 3, 4, 5, 6, 7, 8]
        auto vb = op::sequence(2); // [2, 3, 4, 5, 6, 7, 8, 9]

        // --- 辅助 Lambda：用于计算每 4 个元素的预期点积和 ---
        auto calc_expected_dot = [&](size_t sub_base, int src_mask) {
            FLOAT_T sum = 0;
            if (src_mask & op::X) sum += FLOAT_T(sub_base + 1) * FLOAT_T(sub_base + 2);
            if (src_mask & op::Y) sum += FLOAT_T(sub_base + 2) * FLOAT_T(sub_base + 3);
            if (src_mask & op::Z) sum += FLOAT_T(sub_base + 3) * FLOAT_T(sub_base + 4);
            if (src_mask & op::W) sum += FLOAT_T(sub_base + 4) * FLOAT_T(sub_base + 5);
            return sum;
        };

        // --- Case 1: 测试 src_mask (X|Y) -> dst_mask (All) ---
        // 期望：每 4 个元素计算 X+Y 的点积，并广播到该 4 个位置
        op::store(res, op::dot<op::X | op::Y, op::All>(va, vb));
        for (size_t sub_base = 0; sub_base < TOTAL; sub_base += DOT_UNIT) {
            FLOAT_T expected_sum = calc_expected_dot(sub_base, op::X | op::Y);
            for (size_t l = 0; l < DOT_UNIT; ++l) {
                EXPECT_NEAR(res[sub_base + l], expected_sum, eps)
                    << "Error at Lane " << (sub_base + l) << " (X|Y -> All)";
            }
        }

        // --- Case 2: 测试 dst_mask (仅存入 Z) ---
        // 期望：每 4 个元素计算全点积，结果只出现在索引为 2 的位置（如 res[2], res[6]）
        op::store(res, op::dot<op::All, op::Z>(va, vb));
        for (size_t sub_base = 0; sub_base < TOTAL; sub_base += DOT_UNIT) {
            FLOAT_T expected_sum = calc_expected_dot(sub_base, op::All);
            for (size_t l = 0; l < DOT_UNIT; ++l) {
                FLOAT_T expected = (l == 2) ? expected_sum : FLOAT_T(0);
                EXPECT_NEAR(res[sub_base + l], expected, eps)
                    << "Error at Lane " << (sub_base + l) << " (All -> Z)";
            }
        }

        // --- Case 3: 边界测试 (负数与混合掩码 X|W) ---
        auto vc = op::set(FLOAT_T(-2.5));
        auto vd = op::set(FLOAT_T(4.0));
        op::store(res, op::dot<op::X | op::W, op::X | op::W>(vc, vd));

        // sum = (-2.5 * 4.0) + (-2.5 * 4.0) = -20.0
        for (size_t sub_base = 0; sub_base < TOTAL; sub_base += DOT_UNIT) {
            for (size_t l = 0; l < DOT_UNIT; ++l) {
                FLOAT_T expected = (l == 0 || l == 3) ? FLOAT_T(-20) : FLOAT_T(0);
                EXPECT_NEAR(res[sub_base + l], expected, eps)
                    << "Error at Lane " << (sub_base + l) << " (Negative X|W)";
            }
        }

        // --- Case 4: 零掩码测试 (None) ---
        op::store(res, op::dot<op::None, op::All>(va, vb));
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

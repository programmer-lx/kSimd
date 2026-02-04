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
        using op = KSIMD_DYN_FIXED_OP(FLOAT_T, 4);
        alignas(ALIGNMENT) FLOAT_T res[4]{};

        // 定义一个适合当前精度的阈值
        // dot 包含多次乘加，通常使用 epsilon 的 10 倍左右作为容差
        constexpr FLOAT_T eps = std::numeric_limits<FLOAT_T>::epsilon() * FLOAT_T(10);

        // 准备数据: a = [1, 2, 3, 4], b = [2, 3, 4, 5]
        auto a = op::sequence(1);
        auto b = op::sequence(2);

        // 1. 测试 src_mask: 只有 X, Y 参与计算 -> 1*2 + 2*3 = 8
        op::store(res, op::dot<op::X | op::Y, op::All>(a, b));
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(res[i], FLOAT_T(8), eps);
        }

        // 2. 测试 dst_mask: 全参与 -> 40, 只存入 Z
        op::store(res, op::dot<op::All, op::Z>(a, b));
        EXPECT_NEAR(res[0], FLOAT_T(0), eps);
        EXPECT_NEAR(res[1], FLOAT_T(0), eps);
        EXPECT_NEAR(res[2], FLOAT_T(40), eps);
        EXPECT_NEAR(res[3], FLOAT_T(0), eps);

        // 3. 测试 0 掩码
        op::store(res, op::dot<op::None, op::All>(a, b));
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(res[i], FLOAT_T(0), eps);
        }

        // 4. 边界测试: 负数与较大数值
        auto c = op::set(FLOAT_T(-2.5));
        auto d = op::set(FLOAT_T(4.0));
        // sum = -2.5 * 4.0 * 4 = -40.0
        op::store(res, op::dot<op::All, op::X | op::W>(c, d));
        EXPECT_NEAR(res[0], FLOAT_T(-40), eps);
        EXPECT_NEAR(res[1], FLOAT_T(0), eps);
        EXPECT_NEAR(res[2], FLOAT_T(0), eps);
        EXPECT_NEAR(res[3], FLOAT_T(-40), eps);
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

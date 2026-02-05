#include "../../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/math/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

// 下面这两个文件一定要放在dispatch_this_file.hpp之后
#include <kSimd/base_op.hpp>
#include <kSimd_extension/vmath.hpp>


// ------------------------------------------ sin ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sin() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::TotalLanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        alignas(ALIGNMENT) FLOAT_T in_data[Lanes];
        alignas(ALIGNMENT) FLOAT_T out_data[Lanes];

        // 1. 常规步进测试 (-10.0 to 10.0)
        for (FLOAT_T val = FLOAT_T(-10); val < FLOAT_T(10); val += FLOAT_T(0.5)) {
            for (size_t i = 0; i < Lanes; ++i) {
                in_data[i] = val + FLOAT_T(i) * FLOAT_T(0.01);
            }
            op::store(out_data, ext::vmath::sin<op>(op::load(in_data)));
            for (size_t i = 0; i < Lanes; ++i) {
                if (std::isnan(in_data[i]) || std::isinf(in_data[i])) { // 特殊值处理
                    EXPECT_TRUE(std::isnan(out_data[i]));
                } else {
                    EXPECT_NEAR(out_data[i], std::sin(in_data[i]), std::numeric_limits<FLOAT_T>::epsilon() * 100);
                }
            }
        }

        // 2. 特殊值点对点测试
        const FLOAT_T cases[] = { FLOAT_T(0), FLOAT_T(-0.0), qNaN<FLOAT_T>, inf<FLOAT_T>, -inf<FLOAT_T> };
        for (size_t i = 0; i < Lanes; ++i) {
            in_data[i] = cases[i % 5];
        }
        op::store(out_data, ext::vmath::sin<op>(op::load(in_data)));
        for (size_t i = 0; i < Lanes; ++i) {
            if (std::isnan(in_data[i]) || std::isinf(in_data[i])) { // 指数全 1 判定
                EXPECT_TRUE(std::isnan(out_data[i]));
            } else {
                EXPECT_EQ(out_data[i], std::sin(in_data[i]));
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(sin)
#endif

// ------------------------------------------ lerp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void lerp() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::TotalLanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        alignas(ALIGNMENT) FLOAT_T aa[Lanes], bb[Lanes], tt[Lanes], rr[Lanes];

        // 1. 基础与边界值测试
        #define run_lerp_test(va, vb, vt) \
            do { \
                for (size_t i = 0; i < Lanes; ++i) { aa[i] = va; bb[i] = vb; tt[i] = vt; } \
                op::store(rr, ext::vmath::lerp<op>(op::load(aa), op::load(bb), op::load(tt))); \
            } while (0)

        // 基础插值与外插
        run_lerp_test(0, 2, 0.6f);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(rr[i], 1.2f, std::numeric_limits<FLOAT_T>::epsilon() * 10);

        run_lerp_test(10, 20, 0);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(rr[i], 10);

        run_lerp_test(10, 20, 1);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(rr[i], 20);

        // 2. 特殊值测试
        // Case: NaN 参与运算 (结果必为 NaN)
        run_lerp_test(qNaN<FLOAT_T>, 1.0f, 0.5f); // 阶码全 1, 尾数 != 0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(rr[i]));

        // Case: Inf 参与运算
        run_lerp_test(0, inf<FLOAT_T>, 0.5f); // 阶码全 1, 尾数 == 0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(rr[i]));

        run_lerp_test(0, 1.0f, inf<FLOAT_T>);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(rr[i]));

        #undef run_lerp_test
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(lerp)
#endif


// main function
#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
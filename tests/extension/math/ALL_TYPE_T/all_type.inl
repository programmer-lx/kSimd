#include "../../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/math/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

// 下面这两个文件一定要放在dispatch_this_file.hpp之后
#include <kSimd/op.hpp>
#include <kSimd_extension/vmath.hpp>

// ------------------------------------------ clamp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void clamp() noexcept
    {
        using op = KSIMD_DYN_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        alignas(ALIGNMENT) TYPE_T a[Lanes], b[Lanes], c[Lanes], r[Lanes];

        #define run_test(va, vb, vc) \
        { \
            for (size_t i = 0; i < Lanes; ++i) { a[i] = va; b[i] = vb; c[i] = vc; } \
            op::store(r, ext::vmath::clamp<op>(op::load(a), op::load(b), op::load(c))); \
        }

        // --- 1. Inf 测试 ---
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // 输入为 Inf，被有限边界截断
            run_test(inf<TYPE_T>, TYPE_T(-10), TYPE_T(10));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(r[i], TYPE_T(10));

            // 输入为有限值，边界为 Inf
            run_test(TYPE_T(100), -inf<TYPE_T>, inf<TYPE_T>);
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(r[i], TYPE_T(100));

            // 输入为 -Inf，被有限最小值截断
            run_test(-inf<TYPE_T>, TYPE_T(0), TYPE_T(10));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(r[i], TYPE_T(0));

            // 区间收窄到极点
            run_test(TYPE_T(1), inf<TYPE_T>, inf<TYPE_T>);
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(r[i]) && r[i] > 0);

            // --- 2. NaN 边界测试 (重点验证非传播性) ---
            // 库不处理NaN，不做测试
        }

        // --- 3. 整数/通用基础测试 ---
        run_test(TYPE_T(10), TYPE_T(20), TYPE_T(30)); // < min
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(r[i], TYPE_T(20));

        #undef run_test
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(clamp)
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
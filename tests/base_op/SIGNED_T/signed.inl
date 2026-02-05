#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/SIGNED_T/signed.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/base_op.hpp>

using namespace ksimd;

// ------------------------------------------ abs ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void abs() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 正数与负数
        op::store(test, op::abs(op::set(TYPE_T(-5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(5));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // -0.0 -> 0.0
            op::store(test, op::abs(op::set(TYPE_T(-0.0))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(test[i], TYPE_T(0.0));
                EXPECT_FALSE(std::signbit(test[i])); // 验证符号位已清除
            }

            // -Inf -> Inf
            op::store(test, op::abs(op::set(-inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(abs)
#endif

// ------------------------------------------ neg ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void neg() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T src[Lanes];
        alignas(ALIGNMENT) TYPE_T dst[Lanes];

        // =========================================================================
        // 1. 基础数值测试 (Basic Values)
        // 验证：正变负，负变正
        // =========================================================================
        for (size_t i = 0; i < Lanes; ++i) {
            // 构造 [1, -2, 3, -4, ...]
            src[i] = (i % 2 == 0) ? TYPE_T(i + 1) : TYPE_T(-(static_cast<int>(i) + 1));
        }
        op::store(dst, op::neg(op::load(src)));
        for (size_t i = 0; i < Lanes; ++i) {
            TYPE_T expected = static_cast<TYPE_T>(-src[i]);
            EXPECT_EQ(dst[i], expected) << "Basic negation failed at index " << i;
        }

        // =========================================================================
        // 2. 双重否定测试 (Double Negation)
        // 验证：-(-x) == x
        // =========================================================================
        op::store(dst, op::neg(op::neg(op::load(src))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], src[i]) << "Double negation failed at index " << i;
        }

        // =========================================================================
        // 3. 零值与符号位测试 (Zero & Sign Bit)
        // 对于整数：0 -> 0
        // 对于浮点数：0.0 -> -0.0 (符号位必须翻转)
        // =========================================================================
        for (size_t i = 0; i < Lanes; ++i) src[i] = TYPE_T(0);
        op::store(dst, op::neg(op::load(src)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], TYPE_T(0));
            if constexpr (std::is_floating_point_v<TYPE_T>) {
                // 在 IEEE 754 中，0.0 和 -0.0 相等，但符号位不同
                // 必须验证底层符号位确实翻转了
                EXPECT_NE(std::signbit(src[i]), std::signbit(dst[i])) << "Zero sign flip failed";
            }
        }

        // =========================================================================
        // 4. 整数特有测试：边界与溢出 (Integer Limits)
        // =========================================================================
        if constexpr (std::is_integral_v<TYPE_T>) {
            // A. 最大正值
            for (size_t i = 0; i < Lanes; ++i) src[i] = std::numeric_limits<TYPE_T>::max();
            op::store(dst, op::neg(op::load(src)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(dst[i], static_cast<TYPE_T>(-std::numeric_limits<TYPE_T>::max()));
            }

            // B. 最小负值 (关键：补码中 -INT_MIN 溢出后仍为 INT_MIN)
            for (size_t i = 0; i < Lanes; ++i) src[i] = std::numeric_limits<TYPE_T>::min();
            op::store(dst, op::neg(op::load(src)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(dst[i], std::numeric_limits<TYPE_T>::min()) << "INT_MIN negation overflow failed";
            }
        }

        // =========================================================================
        // 5. 浮点数特有测试：特殊值 (Floating Point Specials)
        // =========================================================================
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // A. 正无穷变负无穷 (Inf -> -Inf)
            for (size_t i = 0; i < Lanes; ++i) src[i] = inf<TYPE_T>;
            op::store(dst, op::neg(op::load(src)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isinf(dst[i]));
                EXPECT_TRUE(std::signbit(dst[i])) << "Inf to -Inf failed";
            }

            // B. 负无穷变正无穷 (-Inf -> Inf)
            for (size_t i = 0; i < Lanes; ++i) src[i] = -inf<TYPE_T>;
            op::store(dst, op::neg(op::load(src)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isinf(dst[i]));
                EXPECT_FALSE(std::signbit(dst[i])) << "-Inf to Inf failed";
            }

            // C. NaN 的符号位翻转 (NaN -> -NaN)
            // 虽然 NaN 无数值意义，但 neg 应该一致地翻转其符号位
            for (size_t i = 0; i < Lanes; ++i) src[i] = qNaN<TYPE_T>;
            op::store(dst, op::neg(op::load(src)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isnan(dst[i]));
                EXPECT_NE(std::signbit(src[i]), std::signbit(dst[i])) << "NaN sign flip failed";
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(neg)
#endif


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
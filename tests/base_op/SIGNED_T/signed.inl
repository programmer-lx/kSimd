#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/SIGNED_T/signed.inl" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>

// ------------------------------------------ abs ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void abs() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 基础数值：正数与负数
        ns::store(test, ns::abs(ns::set(TYPE_T(-5))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(5)));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // -0.0 -> 0.0 (符号位必须清除)
            ns::store(test, ns::abs(ns::set(TYPE_T(-0.0))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(test[i], TYPE_T(0.0));
                EXPECT_FALSE(std::signbit(test[i]));
            }

            // -Inf -> Inf
            ns::store(test, ns::abs(ns::set(-inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);
            }
        }

        if constexpr (std::is_integral_v<TYPE_T> && std::is_signed_v<TYPE_T>) {
            // 补码特例：abs(INT_MIN) 在溢出后仍为 INT_MIN
            ns::store(test, ns::abs(ns::set(std::numeric_limits<TYPE_T>::min())));
            EXPECT_TRUE(array_equal(test, Lanes, std::numeric_limits<TYPE_T>::min()));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T src[Lanes];
        alignas(ns::Alignment<TYPE_T>) TYPE_T dst[Lanes];

        // 1. 基础数值测试：正变负，负变正
        for (size_t i = 0; i < Lanes; ++i) {
            src[i] = (i % 2 == 0) ? TYPE_T(i + 1) : TYPE_T(-(static_cast<int>(i) + 1));
        }
        ns::store(dst, ns::neg(ns::load(src)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], static_cast<TYPE_T>(-src[i]));
        }

        // 2. 双重否定：-(-x) == x
        ns::store(dst, ns::neg(ns::neg(ns::load(src))));
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(dst[i] == src[i]);
        }

        // 3. 零值符号位测试
        ns::store(src, ns::set(TYPE_T(0)));
        ns::store(dst, ns::neg(ns::load(src)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], TYPE_T(0));
            if constexpr (std::is_floating_point_v<TYPE_T>) {
                // IEEE 754: 0.0 -> -0.0
                EXPECT_NE(std::signbit(src[i]), std::signbit(dst[i]));
            }
        }

        // 4. 整数边界测试
        if constexpr (std::is_integral_v<TYPE_T> && std::is_signed_v<TYPE_T>) {
            // INT_MIN -> INT_MIN (溢出行为)
            batch_t v_min = ns::set(std::numeric_limits<TYPE_T>::min());
            ns::store(dst, ns::neg(v_min));
            EXPECT_TRUE(array_equal(dst, Lanes, std::numeric_limits<TYPE_T>::min()));
        }

        // 5. 浮点数特殊值
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Inf -> -Inf
            ns::store(dst, ns::neg(ns::set(inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isinf(dst[i]) && std::signbit(dst[i]));
            }

            // NaN sign flip
            batch_t v_nan = ns::set(qNaN<TYPE_T>);
            ns::store(src, v_nan);
            ns::store(dst, ns::neg(v_nan));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(std::isnan(dst[i]));
                EXPECT_NE(std::signbit(src[i]), std::signbit(dst[i]));
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
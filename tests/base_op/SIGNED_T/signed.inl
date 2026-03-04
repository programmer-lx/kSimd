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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        // 基础数值：正数与负数
        ns::store(t, test, ns::abs(t, ns::set(t, TYPE_T(-5))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(5)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // -0.0 -> 0.0 (符号位必须清除)
            ns::store(t, test, ns::abs(t, ns::set(t, TYPE_T(-0.0))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(test[i], TYPE_T(0.0));
                EXPECT_FALSE(ksimd::sign_bit(test[i]));
            }

            // -Inf -> Inf
            ns::store(t, test, ns::abs(t, ns::set(t, -ksimd::Inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_inf(test[i]) && test[i] > 0);
            }
        }

        // if constexpr (ksimd::is_scalar_signed_integer<TYPE_T>) {
        //     // 补码特例：abs(INT_MIN) 在溢出后仍为 INT_MIN
        //     ns::store(t, test, ns::abs(t, ns::set(t, ksimd::Min<TYPE_T>)));
        //     EXPECT_TRUE(array_equal(test, Lanes, ksimd::Min<TYPE_T>));
        // }
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using batch_t = ns::Batch<decltype(t)>;

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T src[Lanes];
        alignas(ALIGNMENT) TYPE_T dst[Lanes];

        // 1. 基础数值测试：正变负，负变正
        for (size_t i = 0; i < Lanes; ++i) {
            src[i] = (i % 2 == 0) ? TYPE_T(i + 1) : TYPE_T(-(static_cast<int>(i) + 1));
        }
        ns::store(t, dst, ns::neg(t, ns::load(t, src)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], static_cast<TYPE_T>(-src[i]));
        }

        // 2. 双重否定：-(-x) == x
        ns::store(t, dst, ns::neg(t, ns::neg(t, ns::load(t, src))));
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(dst[i] == src[i]);
        }

        // 3. 零值符号位测试
        ns::store(t, src, ns::set(t, TYPE_T(0)));
        ns::store(t, dst, ns::neg(t, ns::load(t, src)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], TYPE_T(0));
            if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
                // IEEE 754: 0.0 -> -0.0
                EXPECT_NE(ksimd::sign_bit(src[i]), ksimd::sign_bit(dst[i]));
            }
        }

        // 4. 整数边界测试
        // if constexpr (ksimd::is_scalar_signed_integer<TYPE_T>) {
        //     // INT_MIN -> INT_MIN (溢出行为)
        //     batch_t v_min = ns::set(t, ksimd::Min<TYPE_T>);
        //     ns::store(t, dst, ns::neg(t, v_min));
        //     EXPECT_TRUE(array_equal(dst, Lanes, ksimd::Min<TYPE_T>));
        // }

        // 5. 浮点数特殊值
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Inf -> -Inf
            ns::store(t, dst, ns::neg(t, ns::set(t, ksimd::Inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_inf(dst[i]) && ksimd::sign_bit(dst[i]));
            }

            // NaN sign flip
            batch_t v_nan = ns::set(t, ksimd::QNaN<TYPE_T>);
            ns::store(t, src, v_nan);
            ns::store(t, dst, ns::neg(t, v_nan));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_NaN(dst[i]));
                EXPECT_NE(ksimd::sign_bit(src[i]), ksimd::sign_bit(dst[i]));
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
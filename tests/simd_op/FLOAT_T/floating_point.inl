// using FLOAT_T = float;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_op/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/simd_op.hpp>


// ------------------------------------------ one_div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void one_div() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 常规数值
        op::store(test, op::one_div(op::set(FLOAT_T(4))));
        for (size_t i = 0; i < Lanes; ++i)
            EXPECT_NEAR(test[i], FLOAT_T(0.25), FLOAT_T_EPSILON_ONE_DIV);

        // 边界：1/Inf = 0, 1/0 = Inf
        op::store(test, op::one_div(op::set(inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(0));

        op::store(test, op::one_div(op::set(FLOAT_T(0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(one_div)
#endif

// ------------------------------------------ sqrt ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sqrt() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        op::store(test, op::sqrt(op::set(FLOAT_T(16))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(test[i], FLOAT_T(4), FLOAT_T_EPSILON);

        // 边界：sqrt(-1) = NaN
        op::store(test, op::sqrt(op::set(FLOAT_T(-1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(sqrt)
#endif

// ------------------------------------------ rsqrt ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void rsqrt() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        op::store(test, op::rsqrt(op::set(FLOAT_T(4))));
        for (size_t i = 0; i < Lanes; ++i)
            EXPECT_NEAR(test[i], FLOAT_T(0.5), FLOAT_T_EPSILON_RSQRT);

        // 边界：rsqrt(0) = Inf
        op::store(test, op::rsqrt(op::set(FLOAT_T(0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(rsqrt)
#endif

// ------------------------------------------ not_greater ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_greater() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 > 2 为假 -> true
        op::test_store_mask(test, op::not_greater(op::set(FLOAT_T(1)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));

        // NaN 无序特性 (NaN > 2 为假) -> true
        op::test_store_mask(test, op::not_greater(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(not_greater)
#endif

// ------------------------------------------ not_greater_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_greater_equal() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 >= 1 为真 -> false
        op::test_store_mask(test, op::not_greater_equal(op::set(FLOAT_T(1)), op::set(FLOAT_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::zero_block<FLOAT_T>));

        // NaN >= 2 为假 -> true
        op::test_store_mask(test, op::not_greater_equal(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(not_greater_equal)
#endif

// ------------------------------------------ not_less ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_less() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 3 < 2 为假 -> true
        op::test_store_mask(test, op::not_less(op::set(FLOAT_T(3)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));

        // NaN < 2 为假 -> true
        op::test_store_mask(test, op::not_less(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(not_less)
#endif

// ------------------------------------------ not_less_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_less_equal() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 <= 2 为真 -> false
        op::test_store_mask(test, op::not_less_equal(op::set(FLOAT_T(1)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::zero_block<FLOAT_T>));

        // NaN <= 2 为假 -> true
        op::test_store_mask(test, op::not_less_equal(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(not_less_equal)
#endif

// ------------------------------------------ any_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void any_NaN() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 正常数值 -> false
        op::test_store_mask(test, op::any_NaN(op::set(FLOAT_T(3)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::zero_block<FLOAT_T>));

        // 含有 NaN -> true
        op::test_store_mask(test, op::any_NaN(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(any_NaN)
#endif

// ------------------------------------------ not_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_NaN() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 两者皆为数值 -> true
        op::test_store_mask(test, op::not_NaN(op::set(FLOAT_T(3)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::one_block<FLOAT_T>));

        // 含有 NaN -> false
        op::test_store_mask(test, op::not_NaN(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::zero_block<FLOAT_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(not_NaN)
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
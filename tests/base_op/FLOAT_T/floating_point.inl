// using FLOAT_T = float;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/base_op.hpp>


// ------------------------------------------ one_div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void one_div() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 > 2 为假 -> true
        op::test_store_mask(test, op::not_greater(op::set(FLOAT_T(1)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));

        // NaN 无序特性 (NaN > 2 为假) -> true
        op::test_store_mask(test, op::not_greater(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 >= 1 为真 -> false
        op::test_store_mask(test, op::not_greater_equal(op::set(FLOAT_T(1)), op::set(FLOAT_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

        // NaN >= 2 为假 -> true
        op::test_store_mask(test, op::not_greater_equal(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 3 < 2 为假 -> true
        op::test_store_mask(test, op::not_less(op::set(FLOAT_T(3)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));

        // NaN < 2 为假 -> true
        op::test_store_mask(test, op::not_less(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 1 <= 2 为真 -> false
        op::test_store_mask(test, op::not_less_equal(op::set(FLOAT_T(1)), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

        // NaN <= 2 为假 -> true
        op::test_store_mask(test, op::not_less_equal(op::set(qNaN<FLOAT_T>), op::set(FLOAT_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
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
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        using uint_t = std::conditional_t<sizeof(FLOAT_T) == 4, uint32_t, uint64_t>;
        const uint_t base_nan = std::bit_cast<uint_t>(qNaN<FLOAT_T>);
        const FLOAT_T n_hi = std::bit_cast<FLOAT_T>(base_nan | (uint_t(1) << (std::numeric_limits<FLOAT_T>::digits - 2)));
        const FLOAT_T n_lo = std::bit_cast<FLOAT_T>(base_nan | uint_t(1));

#define run_any(a, b) do { op::test_store_mask(test, op::any_NaN(op::set(a), op::set(b))); } while (0)

        run_any(FLOAT_T(1.0), FLOAT_T(2.0));      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_any(inf<FLOAT_T>, -inf<FLOAT_T>);     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_any(n_hi, FLOAT_T(0.0));              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(FLOAT_T(-1.0), n_lo);             EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(n_hi, n_lo);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(n_lo, inf<FLOAT_T>);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(qNaN<FLOAT_T>, qNaN<FLOAT_T>);    EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(FLOAT_T(0.0), FLOAT_T(0.0));      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

#undef run_any
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(any_NaN)
#endif

// ------------------------------------------ all_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void all_NaN() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        using uint_t = std::conditional_t<sizeof(FLOAT_T) == 4, uint32_t, uint64_t>;
        const uint_t base_nan = std::bit_cast<uint_t>(qNaN<FLOAT_T>);
        const FLOAT_T n_hi = std::bit_cast<FLOAT_T>(base_nan | (uint_t(1) << (std::numeric_limits<FLOAT_T>::digits - 2)));
        const FLOAT_T n_lo = std::bit_cast<FLOAT_T>(base_nan | uint_t(1));

#define run_all(a, b) do { op::test_store_mask(test, op::all_NaN(op::set(a), op::set(b))); } while (0)

        run_all(n_hi, n_lo);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_all(n_lo, n_hi);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_all(n_hi, n_hi);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_all(n_hi, inf<FLOAT_T>);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(inf<FLOAT_T>, n_lo);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(FLOAT_T(1.0), n_hi);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(n_lo, FLOAT_T(0.0));              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(inf<FLOAT_T>, -inf<FLOAT_T>);     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

#undef run_all
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(all_NaN)
#endif

// ------------------------------------------ not_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_NaN() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        using uint_t = std::conditional_t<sizeof(FLOAT_T) == 4, uint32_t, uint64_t>;
        const uint_t base_nan = std::bit_cast<uint_t>(qNaN<FLOAT_T>);
        const FLOAT_T n_hi = std::bit_cast<FLOAT_T>(base_nan | (uint_t(1) << (std::numeric_limits<FLOAT_T>::digits - 2)));
        const FLOAT_T n_lo = std::bit_cast<FLOAT_T>(base_nan | uint_t(1));

#define run_not(a, b) do { op::test_store_mask(test, op::not_NaN(op::set(a), op::set(b))); } while (0)

        run_not(FLOAT_T(1.0), FLOAT_T(2.0));      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_not(inf<FLOAT_T>, -inf<FLOAT_T>);     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_not(FLOAT_T(0.0), inf<FLOAT_T>);      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_not(n_hi, FLOAT_T(1.0));              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_not(FLOAT_T(1.0), n_lo);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_not(n_hi, n_lo);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_not(n_lo, inf<FLOAT_T>);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

#undef run_not
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(not_NaN)
#endif

// ------------------------------------------ any_finite ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void any_finite() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        using uint_t = std::conditional_t<sizeof(FLOAT_T) == 4, uint32_t, uint64_t>;
        const uint_t base_nan = std::bit_cast<uint_t>(qNaN<FLOAT_T>);
        const FLOAT_T n_hi = std::bit_cast<FLOAT_T>(base_nan | (uint_t(1) << (std::numeric_limits<FLOAT_T>::digits - 2)));
        const FLOAT_T n_lo = std::bit_cast<FLOAT_T>(base_nan | uint_t(1));

#define run_any(a, b) do { op::test_store_mask(test, op::any_finite(op::set(a), op::set(b))); } while (0)

        run_any(FLOAT_T(1.0), n_hi);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(inf<FLOAT_T>, FLOAT_T(0.0));      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(n_lo, FLOAT_T(-1.0));             EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_any(n_hi, n_lo);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_any(inf<FLOAT_T>, n_hi);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_any(-inf<FLOAT_T>, inf<FLOAT_T>);     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_any(qNaN<FLOAT_T>, inf<FLOAT_T>);     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

        if constexpr (std::is_same_v<FLOAT_T, float>) {
            FLOAT_T denorm = std::bit_cast<float>(0x00000001u);
            run_any(denorm, n_hi);                EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        }

#undef run_any
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(any_finite)
#endif

// ------------------------------------------ all_finite ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void all_finite() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        using uint_t = ksimd::same_bits_uint_t<FLOAT_T>;
        const uint_t base_nan_bits = std::bit_cast<uint_t>(qNaN<FLOAT_T>);
        const FLOAT_T n_hi = make_var_from_bits<FLOAT_T>(base_nan_bits | (uint_t(1) << (std::numeric_limits<FLOAT_T>::digits - 2)));

        #define run_all(a, b) do { op::test_store_mask(test, op::all_finite(op::set(a), op::set(b))); } while (0)

        run_all(FLOAT_T(1.0), FLOAT_T(2.0));      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_all(FLOAT_T(0.0), FLOAT_T(-0.0));     EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        run_all(n_hi, FLOAT_T(1.0));              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(FLOAT_T(1.0), inf<FLOAT_T>);      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(inf<FLOAT_T>, n_hi);              EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(inf<FLOAT_T>, inf<FLOAT_T>);      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
        run_all(n_hi, n_hi);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

        if constexpr (sizeof(FLOAT_T) == 4) {
            [[maybe_unused]] const FLOAT_T f1 = make_var_from_bits<FLOAT_T>(0x3F800000u);
            [[maybe_unused]] const FLOAT_T f2 = make_var_from_bits<FLOAT_T>(0x40000000u);
            run_all(f1, f2);                      EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));
        }

        #undef run_all
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(all_finite)
#endif

// ------------------------------------------ round_down ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_down() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 常规：正数向下取整
        op::store(test, op::round<ksimd::RoundingMode::Down>(op::set(FLOAT_T(2.7))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(2.0));

        // 常规：负数向下取整 (floor(-2.1) = -3)
        op::store(test, op::round<ksimd::RoundingMode::Down>(op::set(FLOAT_T(-2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(-3.0));

        // 边界：Inf 和 NaN
        op::store(test, op::round<ksimd::RoundingMode::Down>(op::set(inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

        op::store(test, op::round<ksimd::RoundingMode::Down>(op::set(qNaN<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_down)
#endif

// ------------------------------------------ round_up ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_up() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 常规：正数向上取整
        op::store(test, op::round<ksimd::RoundingMode::Up>(op::set(FLOAT_T(2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(3.0));

        // 常规：负数向上取整 (ceil(-2.7) = -2)
        op::store(test, op::round<ksimd::RoundingMode::Up>(op::set(FLOAT_T(-2.7))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(-2.0));

        // 边界：大数值 (超过有效尾数范围应保持原样)
        FLOAT_T big_val = FLOAT_T(1LL << (std::is_same_v<FLOAT_T, float> ? 25 : 54));
        op::store(test, op::round<ksimd::RoundingMode::Up>(op::set(big_val + FLOAT_T(0.5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], big_val + FLOAT_T(0.5));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_up)
#endif

// ------------------------------------------ round_to_zero ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_to_zero() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        #define check(input, expected, msg) \
            do { \
                op::store(test, op::round<ksimd::RoundingMode::ToZero>(op::set(input))); \
                for (size_t i = 0; i < Lanes; ++i) { \
                    if (std::isnan(expected)) { \
                        EXPECT_TRUE(std::isnan(test[i])) << msg << " | Index: " << i; \
                    } else { \
                        EXPECT_EQ(test[i], expected) << msg << " | Index: " << i; \
                        EXPECT_EQ(std::signbit(test[i]), std::signbit(expected)) \
                                << msg << " Sign bit mismatch | Index: " << i; \
                    } \
                } \
            } while (0)

        // 1. 常规小数 (Normal small values)
        check(FLOAT_T(2.9),   FLOAT_T(2.0),  "Positive truncate");
        check(FLOAT_T(-2.9),  FLOAT_T(-2.0), "Negative truncate");
        check(FLOAT_T(0.1),   FLOAT_T(0.0),  "Positive near zero");
        check(FLOAT_T(-0.1),  FLOAT_T(-0.0), "Negative near zero");

        // 2. 边界值：零与极小值
        check(FLOAT_T(0.0),   FLOAT_T(0.0),  "Zero");
        check(FLOAT_T(-0.0),  FLOAT_T(-0.0), "Negative zero");
        check(std::numeric_limits<FLOAT_T>::denorm_min(), FLOAT_T(0.0), "Denormal positive");

        // 3. 精度分水岭 (2^23): 超过这个值 float 无法表示小数
        // 此时 truncate 结果应为原值
        FLOAT_T pow23 = FLOAT_T(std::numeric_limits<FLOAT_T>::max());
        check(pow23, pow23, "At 2^23 boundary"); // 注意：pow23+0.5 实际上在 float 里就是 pow23

        // 4. SSE2 关键分水岭 (2^31): 硬件 cvttps 溢出的边界
        // 我们需要确保逻辑能跳过硬件溢出，返回正确的大数结果
        FLOAT_T big = FLOAT_T(std::numeric_limits<FLOAT_T>::max() - 20); // 2^31
        check(big + FLOAT_T(10.0), big + FLOAT_T(10.0), "Beyond int32 range (Positive)");
        check(-big - FLOAT_T(10.0), -big - FLOAT_T(10.0), "Beyond int32 range (Negative)");

        // 5. 极大值 (Very large numbers)
        check(FLOAT_T(1e18), FLOAT_T(1e18), "Large exponent value");
        check(std::numeric_limits<FLOAT_T>::max(), std::numeric_limits<FLOAT_T>::max(), "FLT_MAX");

        // 6. 特殊浮点值 (NaN / Inf)
        check(std::numeric_limits<FLOAT_T>::infinity(),
              std::numeric_limits<FLOAT_T>::infinity(), "Positive Infinity");
        check(-std::numeric_limits<FLOAT_T>::infinity(),
              -std::numeric_limits<FLOAT_T>::infinity(), "Negative Infinity");
        check(std::numeric_limits<FLOAT_T>::quiet_NaN(),
              std::numeric_limits<FLOAT_T>::quiet_NaN(), "NaN");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_to_zero)
#endif

// ------------------------------------------ round_nearest ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_nearest() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // 最近舍入 (Ties to Even 验证)
        // 2.5 -> 2.0 (最近偶数)
        op::store(test, op::round<ksimd::RoundingMode::Nearest>(op::set(FLOAT_T(2.5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(2.0));

        // 3.5 -> 4.0 (最近偶数)
        op::store(test, op::round<ksimd::RoundingMode::Nearest>(op::set(FLOAT_T(3.5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(4.0));

        // 常规数值
        op::store(test, op::round<ksimd::RoundingMode::Nearest>(op::set(FLOAT_T(-2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(-2.0));

        op::store(test, op::round<ksimd::RoundingMode::Nearest>(op::set(FLOAT_T(-2.9))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(-3.0));

        // 边界：NaN
        op::store(test, op::round<ksimd::RoundingMode::Nearest>(op::set(qNaN<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_nearest)
#endif

// ------------------------------------------ round ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) FLOAT_T test[Lanes]{};

        // --- 1. 标准四舍五入 (Rounding away from zero) ---
        // 正数 0.5 进位
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(2.5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(3.0));

        // 负数 0.5 进位 (远离零方向)
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(-2.5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(-3.0));


        // --- 2. 临界值测试 (Precision Boundary) ---
        // 使用 epsilon 确保测试的是该类型能表示的最小差异
        const FLOAT_T eps = std::numeric_limits<FLOAT_T>::epsilon();

        // 测试略小于 0.5 的情况 (应舍向 0)
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(0.5) - eps)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(0.0));

        // 测试略大于 0.5 的情况 (应舍向 1)
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(0.5) + eps)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], FLOAT_T(1.0));


        // --- 3. 符号位与零 (Signbit preservation) ---
        // -0.0 经过 round 后依然应该是 -0.0
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(-0.0))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test[i], FLOAT_T(0.0));
            EXPECT_TRUE(std::signbit(test[i]));
        }

        // 负的小数 (未达进位点) 应该保持负零符号
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(FLOAT_T(-0.1))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test[i], FLOAT_T(0.0));
            EXPECT_TRUE(std::signbit(test[i]));
        }


        // --- 4. 大数值测试 (Integer range overflow) ---
        // 跨越有效尾数范围的边界测试
        FLOAT_T big_val;
        if constexpr (std::is_same_v<FLOAT_T, float>) {
            big_val = FLOAT_T(8388608.0); // 2^23: float 精度消失点
        } else {
            big_val = FLOAT_T(4503599627370496.0); // 2^52: double 精度消失点
        }

        // 大值本身就是整数，round 之后不应改变
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(big_val)));
        for (size_t i = 0; i < Lanes; ++i) FLOAT_T_EQ(test[i], big_val);

        // 大值 + 1.0 同样
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(big_val + FLOAT_T(1.0))));
        for (size_t i = 0; i < Lanes; ++i) FLOAT_T_EQ(test[i], big_val + FLOAT_T(1.0));


        // --- 5. 特殊值处理 ---
        // 正负无穷
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]));

        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(-inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_TRUE(std::isinf(test[i]));
            EXPECT_TRUE(std::signbit(test[i]));
        }

        // NaN 处理
        op::store(test, op::round<ksimd::RoundingMode::Round>(op::set(qNaN<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round)
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
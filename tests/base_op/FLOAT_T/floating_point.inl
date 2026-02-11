// using FLOAT_T = float;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/FLOAT_T/floating_point.inl" // this file


#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>

KSIMD_WARNING_PUSH
KSIMD_IGNORE_WARNING_MSVC(4723) // ignore warning: divide by 0

// ------------------------------------------ rcp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void rcp() noexcept
    {
        if constexpr (ksimd::is_scalar_type_float_32bits<FLOAT_T>)
        {
            namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
            
            constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
            alignas(ns::Alignment<FLOAT_T>) FLOAT_T test[Lanes];

            EXPECT_TRUE(std::isinf(inf<FLOAT_T>));
            EXPECT_TRUE(std::isinf(-inf<FLOAT_T>));

            EXPECT_TRUE(ksimd::is_inf(inf<FLOAT_T>));
            EXPECT_TRUE(ksimd::is_inf(-inf<FLOAT_T>));

            // 1. 常规数值 1/4 = 0.25
            ns::store(test, ns::rcp(ns::set(FLOAT_T(4))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_NEAR(static_cast<double>(test[i]), 0.25, FLOAT_T_EPSILON_RCP);
            }

            // 2. 边界：1/Inf = 0
            ns::store(test, ns::rcp(ns::set(inf<FLOAT_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_EQ(test[i], FLOAT_T(0));
                EXPECT_TRUE(!std::signbit(test[i]));
            }

            // 1 / -Inf = -0
            ns::store(test, ns::rcp(ns::set(-inf<FLOAT_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_EQ(test[i], FLOAT_T(-0));
                EXPECT_TRUE(std::signbit(test[i]));
            }

            // 1/0 = Inf
            ns::store(test, ns::rcp(ns::set(FLOAT_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(test[i] == inf<FLOAT_T>);

            // 1/-0 = Inf
            ns::store(test, ns::rcp(ns::set(FLOAT_T(-0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(test[i] == inf<FLOAT_T>);

            // 3. 1/NaN = NaN
            ns::store(test, ns::rcp(ns::set(qNaN<FLOAT_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            ns::store(test, ns::rcp(ns::set(-qNaN<FLOAT_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }

// ------------------------------------------ sqrt ------------------------------------------
    KSIMD_DYN_FUNC_ATTR
    void sqrt() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T test[Lanes];

        ns::store(test, ns::sqrt(ns::set(FLOAT_T(16))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(static_cast<double>(test[i]), 4.0, 1e-7);

        // 负数开方为 NaN
        ns::store(test, ns::sqrt(ns::set(FLOAT_T(-1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
    }

// ------------------------------------------ rsqrt ------------------------------------------
    KSIMD_DYN_FUNC_ATTR
    void rsqrt() noexcept
    {
        if constexpr (ksimd::is_scalar_type_float_32bits<FLOAT_T>)
        {
            namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
            
            constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
            alignas(ns::Alignment<FLOAT_T>) FLOAT_T test[Lanes];

            ns::store(test, ns::rsqrt(ns::set(FLOAT_T(4))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(test[i], 0.5, FLOAT_T_EPSILON_RSQRT);

            ns::store(test, ns::rsqrt(ns::set(FLOAT_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]));

            ns::store(test, ns::rsqrt(ns::set(FLOAT_T(-0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]));

            ns::store(test, ns::rsqrt(ns::set(FLOAT_T(-2))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(rcp)
TEST_ONCE_DYN(sqrt)
TEST_ONCE_DYN(rsqrt)
#endif

// ------------------------------------------ float_not_comparison ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void float_not_comparison(size_t idx) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T res_normal[Lanes];
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T res_not[Lanes];

        // 只有浮点数需要验证 NaN 的特殊取反逻辑
        if constexpr (std::is_floating_point_v<FLOAT_T>) 
        {
            auto v_nan = ns::set(qNaN<FLOAT_T>);
            auto v_val = ns::set(FLOAT_T(2.0));

            // =========================================================================
            // 1. Greater vs Not Greater
            // =========================================================================
            ns::test_store_mask(res_normal, ns::greater(v_nan, v_val));
            ns::test_store_mask(res_not,    ns::not_greater(v_nan, v_val));

            for (size_t i = 0; i < Lanes; ++i) {
                // NaN > 2.0 -> False
                EXPECT_TRUE(bit_equal(res_normal[i], ksimd::ZeroBlock<FLOAT_T>));
                // NOT (NaN > 2.0) -> True (即使是无序比较，逻辑上也必须相反)
                EXPECT_TRUE(bit_equal(res_not[i], ksimd::OneBlock<FLOAT_T>));
            }

            // =========================================================================
            // 2. Less Equal vs Not Less Equal
            // =========================================================================
            ns::test_store_mask(res_normal, ns::less_equal(v_nan, v_val));
            ns::test_store_mask(res_not,    ns::not_less_equal(v_nan, v_val));

            for (size_t i = 0; i < Lanes; ++i) {
                // NaN <= 2.0 -> False
                EXPECT_TRUE(bit_equal(res_normal[i], ksimd::ZeroBlock<FLOAT_T>))
                << "idx = " << idx
                << ", value = " << std::bit_cast<ksimd::same_bits_uint_t<FLOAT_T>>(res_not[i]);

                // NOT (NaN <= 2.0) -> True
                EXPECT_TRUE(bit_equal(res_not[i], ksimd::OneBlock<FLOAT_T>))
                << "idx = " << idx
                << ", value = " << std::bit_cast<ksimd::same_bits_uint_t<FLOAT_T>>(res_not[i]);
            }

            // =========================================================================
            // 3. 验证与“相反操作”的差异
            // 关键：not_greater(NaN, val) != less_equal(NaN, val)
            // =========================================================================
            ns::test_store_mask(res_normal, ns::less_equal(v_nan, v_val));
            // 此时 res_not 存的是 not_greater 的结果 (True)
            // 而 res_normal 存的是 less_equal 的结果 (False)
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_FALSE(bit_equal(res_normal[i], res_not[i])) 
                    << "NaN ordering logic failure: not_greater should not be same as less_equal";
            }
        }
    }
}
#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(float_not_comparison);
TEST(dyn_dispatch, float_not_comparison)
{
    for (size_t idx___ = 0; idx___ < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(float_not_comparison)); ++idx___)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(float_not_comparison)[idx___](idx___);
    }
}
#endif

// ------------------------------------------ nan_finite_checks ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void nan_finite_checks() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T test[Lanes];

        // any_NaN: 只要有一个是 NaN 就返回 True
        ns::test_store_mask(test, ns::any_NaN(ns::set(qNaN<FLOAT_T>), ns::set(FLOAT_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<FLOAT_T>));

        // all_NaN: 两个都是 NaN 才返回 True
        ns::test_store_mask(test, ns::all_NaN(ns::set(qNaN<FLOAT_T>), ns::set(FLOAT_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));

        // all_finite: 两个都必须是有限数
        ns::test_store_mask(test, ns::all_finite(ns::set(FLOAT_T(1)), ns::set(inf<FLOAT_T>)));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<FLOAT_T>));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(nan_finite_checks)
#endif

// ------------------------------------------ round_ops ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_ops() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T res[Lanes];

        // 1. Round Down (Floor)
        ns::store(res, ns::round<ksimd::RoundingMode::Down>(ns::set(FLOAT_T(-2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(-3.0));

        // 2. Round Up (Ceil)
        ns::store(res, ns::round<ksimd::RoundingMode::Up>(ns::set(FLOAT_T(2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(3.0));

        // 3. Round To Zero (Truncate)
        ns::store(res, ns::round<ksimd::RoundingMode::ToZero>(ns::set(FLOAT_T(-2.9))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(-2.0));

        // 4. Round Nearest (Ties to Even)
        ns::store(res, ns::round<ksimd::RoundingMode::Nearest>(ns::set(FLOAT_T(2.5)))); // 2.5 -> 2.0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(2.0));
        ns::store(res, ns::round<ksimd::RoundingMode::Nearest>(ns::set(FLOAT_T(3.5)))); // 3.5 -> 4.0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(4.0));

        // 5. Round (Away from Zero)
        ns::store(res, ns::round<ksimd::RoundingMode::Round>(ns::set(FLOAT_T(-2.5)))); // -2.5 -> -3.0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(-3.0));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_ops)
#endif

// ------------------------------------------ round_edge_cases ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_edge_cases() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T res[Lanes];

        // 验证 -0.0 的符号位在 Round 后保留
        ns::store(res, ns::round<ksimd::RoundingMode::Round>(ns::set(FLOAT_T(-0.0))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(res[i], FLOAT_T(0.0));
            EXPECT_TRUE(std::signbit(res[i]));
        }

        // 验证超大整数（超过尾数精度范围）不被舍入破坏
        // 对于 float，2^24 以上本身就是整数
        FLOAT_T big_val = std::pow(FLOAT_T(2), FLOAT_T(std::numeric_limits<FLOAT_T>::digits) + FLOAT_T(1));
        ns::store(res, ns::round<ksimd::RoundingMode::Up>(ns::set(big_val)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], big_val);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(round_edge_cases)
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

KSIMD_WARNING_POP

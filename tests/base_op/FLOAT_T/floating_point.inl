// using FLOAT_T = float;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/FLOAT_T/floating_point.inl" // this file


#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/core/aligned_allocate.hpp>
#include <vector>

KSIMD_WARNING_PUSH
KSIMD_IGNORE_WARNING_MSVC(4723) // ignore warning: divide by 0


// ------------------------------------------ div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void div() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> test(Lanes);

        // 常规数值测试
        ns::store(t, test.data(), ns::div(t, ns::set(t, FLOAT_T(100)), ns::set(t, FLOAT_T(4))));
        for (size_t i = 0; i < Lanes; ++i) {
            // 使用标准 EXPECT_NEAR 验证除法精度
            EXPECT_NEAR(static_cast<double>(test[i]), 25.0, FLOAT_T_EPSILON * 10);
        }

        if constexpr (ksimd::is_scalar_floating_point<FLOAT_T>)
        {
            // 1.0 / 0.0 = Inf
            ns::store(t, test.data(), ns::div(t, ns::set(t, FLOAT_T(1)), ns::set(t, FLOAT_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_inf(test[i]) && test[i] > 0);

            // 0.0 / 0.0 = NaN
            ns::store(t, test.data(), ns::div(t, ns::set(t, FLOAT_T(0)), ns::set(t, FLOAT_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));

            // Inf / Inf = NaN
            ns::store(t, test.data(), ns::div(t, ns::set(t, ksimd::Inf<FLOAT_T>), ns::set(t, ksimd::Inf<FLOAT_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(div)
#endif

// ------------------------------------------ rcp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    // ------------------------------------------ sqrt ------------------------------------------
    KSIMD_DYN_FUNC_ATTR
    void sqrt() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> test(Lanes);

        ns::store(t, test.data(), ns::sqrt(t, ns::set(t, FLOAT_T(16))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(static_cast<double>(test[i]), 4.0, FLOAT_T_EPSILON * 10);

        // 负数开方为 NaN
        ns::store(t, test.data(), ns::sqrt(t, ns::set(t, FLOAT_T(-1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));
    }

#if KSIMD_TEST_F32
    KSIMD_DYN_FUNC_ATTR
    void rcp() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> test(Lanes);

        EXPECT_TRUE(ksimd::is_inf(ksimd::Inf<FLOAT_T>));
        EXPECT_TRUE(ksimd::is_inf(-ksimd::Inf<FLOAT_T>));

        EXPECT_TRUE(ksimd::is_inf(ksimd::Inf<FLOAT_T>));
        EXPECT_TRUE(ksimd::is_inf(-ksimd::Inf<FLOAT_T>));

        // 1. 常规数值 1/4 = 0.25
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, FLOAT_T(4))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_NEAR(static_cast<double>(test[i]), 0.25, FLOAT_T_EPSILON_RCP);
        }

        // 2. 边界：1/Inf = 0
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, ksimd::Inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(test[i], FLOAT_T(0));
            EXPECT_TRUE(!ksimd::sign_bit(test[i]));
        }

        // 1 / -Inf = -0
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, -ksimd::Inf<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(test[i], FLOAT_T(-0));
            EXPECT_TRUE(ksimd::sign_bit(test[i]));
        }

        // 1/0 = Inf
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, FLOAT_T(0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(test[i] == ksimd::Inf<FLOAT_T>);

        // 1/-0 = Inf
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, FLOAT_T(-0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(test[i] == ksimd::Inf<FLOAT_T>);

        // 3. 1/NaN = NaN
        ns::store(t, test.data(), ns::rcp(t, ns::set(t, ksimd::QNaN<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));

        ns::store(t, test.data(), ns::rcp(t, ns::set(t, -ksimd::QNaN<FLOAT_T>)));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));
    }
#endif

// ------------------------------------------ rsqrt ------------------------------------------
#if KSIMD_TEST_F32
    KSIMD_DYN_FUNC_ATTR
    void rsqrt() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> test(Lanes);

        ns::store(t, test.data(), ns::rsqrt(t, ns::set(t, FLOAT_T(4))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(test[i], 0.5, FLOAT_T_EPSILON_RSQRT);

        ns::store(t, test.data(), ns::rsqrt(t, ns::set(t, FLOAT_T(0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_inf(test[i]));

        ns::store(t, test.data(), ns::rsqrt(t, ns::set(t, FLOAT_T(-0))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_inf(test[i]));

        ns::store(t, test.data(), ns::rsqrt(t, ns::set(t, FLOAT_T(-2))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(ksimd::is_NaN(test[i]));
    }
#endif // F32
}
#if KSIMD_ONCE
TEST_ONCE_DYN(sqrt)

#if KSIMD_TEST_F32
    TEST_ONCE_DYN(rcp)
    TEST_ONCE_DYN(rsqrt)
#endif

#endif

// ------------------------------------------ float_not_comparison ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void float_not_comparison() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        auto v_nan = ns::set(t, ksimd::QNaN<FLOAT_T>);
        auto v_val = ns::set(t, FLOAT_T(2.0));

        // =========================================================================
        // Greater vs Not Greater
        // =========================================================================
        auto res = ns::mask_none(t, ns::greater(t, v_nan, v_val));
        EXPECT_TRUE(res);

        res = ns::mask_all(t, ns::not_greater(t, v_nan, v_val));
        EXPECT_TRUE(res);

        // =========================================================================
        // Greater Equal vs Not Greater Equal
        // =========================================================================
        res = ns::mask_none(t, ns::greater_equal(t, v_nan, v_val));
        EXPECT_TRUE(res);

        res = ns::mask_all(t, ns::not_greater_equal(t, v_nan, v_val));
        EXPECT_TRUE(res);

        // =========================================================================
        // Less Equal vs Not Less Equal
        // =========================================================================
        res = ns::mask_none(t, ns::less(t, v_nan, v_val));
        EXPECT_TRUE(res);

        res = ns::mask_all(t, ns::not_less(t, v_nan, v_val));
        EXPECT_TRUE(res);

        // =========================================================================
        // Less Equal vs Not Less Equal
        // =========================================================================
        res = ns::mask_none(t, ns::less_equal(t, v_nan, v_val));
        EXPECT_TRUE(res);

        res = ns::mask_all(t, ns::not_less_equal(t, v_nan, v_val));
        EXPECT_TRUE(res);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(float_not_comparison)
#endif

// ------------------------------------------ nan_finite_checks ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void nan_finite_checks(size_t index) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        TAG_T t;
        

        // any_NaN: 只要有一个是 NaN 就返回 True
        auto res = ns::mask_all(t, ns::any_NaN(t,ns::set(t, ksimd::QNaN<FLOAT_T>), ns::set(t, FLOAT_T(1))));
        EXPECT_TRUE(res) << "idx: " << index;

        // all_NaN: 两个都是 NaN 才返回 True
        res = ns::mask_none(t, ns::all_NaN(t,ns::set(t, ksimd::QNaN<FLOAT_T>), ns::set(t, FLOAT_T(1))));
        EXPECT_TRUE(res) << "idx: " << index;

        // all_finite: 两个都必须是有限数
        res = ns::mask_none(t, ns::all_finite(t,ns::set(t, FLOAT_T(1)), ns::set(t, ksimd::Inf<FLOAT_T>)));
        EXPECT_TRUE(res);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN_WITH_IDX(nan_finite_checks)
#endif

// ------------------------------------------ round_ops ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void round_ops() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> res(Lanes);

        // 1. Round Down (Floor)
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Down>(t,ns::set(t, FLOAT_T(-2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(-3.0));

        // 2. Round Up (Ceil)
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Up>(t,ns::set(t, FLOAT_T(2.1))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(3.0));

        // 3. Round To Zero (Truncate)
        ns::store(t, res.data(), ns::round<ns::RoundingMode::ToZero>(t,ns::set(t, FLOAT_T(-2.9))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(-2.0));

        // 4. Round Nearest (Ties to Even)
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Nearest>(t,ns::set(t, FLOAT_T(2.5)))); // 2.5 -> 2.0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(2.0));
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Nearest>(t,ns::set(t, FLOAT_T(3.5)))); // 3.5 -> 4.0
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], FLOAT_T(4.0));

        // 5. Round (Away from Zero)
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Round>(t,ns::set(t, FLOAT_T(-2.5)))); // -2.5 -> -3.0
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        std::vector<FLOAT_T, ksimd::AlignedAllocator<FLOAT_T>> res(Lanes);

        // 验证 -0.0 的符号位在 Round 后保留
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Round>(t,ns::set(t, FLOAT_T(-0.0))));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(res[i], FLOAT_T(0.0));
            EXPECT_TRUE(ksimd::sign_bit(res[i]));
        }

        // 验证超大整数（超过尾数精度范围）不被舍入破坏
        using uint_t = ksimd::same_bits_uint_t<FLOAT_T>;
        FLOAT_T big_val = static_cast<FLOAT_T>( (uint_t(1) << ksimd::Digits<FLOAT_T>) + 2 );
        ns::store(t, res.data(), ns::round<ns::RoundingMode::Up>(t,ns::set(t, big_val)));
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

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/math/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

// 下面这两个文件一定要放在dispatch_this_file.hpp之后
#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>


// ------------------------------------------ sin ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sin(FLOAT_T x, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        using traits = op::traits;
        constexpr size_t Lanes = traits::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        size_t i = 0;
        for (; i + Lanes <= TOTAL; i += Lanes)
        {
            op::storeu(out + i, ext::math::sin(op::set(x)));
        }
        for (; i < TOTAL; ++i)
        {
            out[i] = ext::math::sin(x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(sin);

TEST(dyn_dispatch_FLOAT_T, sin)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sin)); ++idx)
    {
        FLOAT_T n = FLOAT_C(-10.0);
        for (int c = 0; c < 100; ++c)
        {
            alignas(ALIGNMENT) FLOAT_T out[TOTAL];

            for (size_t i = 0; i < TOTAL; ++i) out[i] = FLOAT_C(-100.0);

            KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sin)[idx](n, out);

            for (size_t i = 0; i < TOTAL; ++i)
            {
                EXPECT_NEAR(out[i], std::sin(n), FLOAT_T_EPSILON);
            }

            n += FLOAT_C(0.0037);
        }
    }
}
#endif

// ------------------------------------------ lerp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void lerp(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        const FLOAT_T* KSIMD_RESTRICT t,
              FLOAT_T* KSIMD_RESTRICT out
    ) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;
        using batch_t = op::batch_t;

        size_t i = 0;
        for (; i + Lanes <= TOTAL; i += Lanes)
        {
            batch_t x = ext::math::lerp(op::load(a + i), op::load(b + i), op::load(t + i));
            op::store(out + i, x);
        }
        for (; i < TOTAL; ++i)
        {
            out[i] = ext::math::lerp(a[i], b[i], t[i]);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(lerp);

TEST(dyn_dispatch_FLOAT_T, lerp)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T t[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // in
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(t, FLOAT_C(0.6));
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), 1.2, FLOAT_T_EPSILON));

        // in & inv a b
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(t, FLOAT_C(0.6));
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), 0.8, FLOAT_T_EPSILON));

        // < 0
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(t, -1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), -2, FLOAT_T_EPSILON));

        // < 0 & inv a b
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(t, -1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), 4, FLOAT_T_EPSILON));

        // > 1
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(t, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), 4, FLOAT_T_EPSILON));

        // > 1 & inv a b
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(t, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), -2, FLOAT_T_EPSILON));
    }
}
#endif

// ------------------------------------------ safe_clamp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void safe_clamp(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        const FLOAT_T* KSIMD_RESTRICT c,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        size_t i = 0;
        for (; i + Lanes <= TOTAL; i += Lanes)
        {
            using batch_t = op::batch_t;

            batch_t x = ext::math::safe_clamp(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, x);
        }
        for (; i < TOTAL; ++i)
        {
            out[i] = ext::math::safe_clamp(a[i], b[i], c[i]);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(safe_clamp);

TEST(dyn_dispatch_FLOAT_T, safe_clamp)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // in
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & min == max
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & unsafe
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 0);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // <
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & min == max
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & unsafe
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // <=
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // <= & unsafe
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // >
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
        // > & min == max
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
        // > & unsafe
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // >=
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
        // >= & unsafe
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(safe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
    }
}
#endif

// ------------------------------------------ clamp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void clamp(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        const FLOAT_T* KSIMD_RESTRICT c,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        size_t i = 0;
        for (; i + Lanes <= TOTAL; i += Lanes)
        {
            using batch_t = op::batch_t;

            batch_t x = ext::math::clamp(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, x);
        }
        for (; i < TOTAL; ++i)
        {
            out[i] = ext::math::clamp(a[i], b[i], c[i]);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(clamp);

TEST(dyn_dispatch_FLOAT_T, clamp)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // in
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & min == max
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & unsafe
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 0);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_FALSE(array_equal(r, std::size(r), 1));

        // <
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & min == max
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // <=
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // >
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // > & min == max
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // >=
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
    }
}
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
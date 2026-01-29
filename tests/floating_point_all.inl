// using FLOAT_T = float;

#include "test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "floating_point_all.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/simd_op.hpp>

// ------------------------------------------ zero ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void zero(FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);

        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::zero());
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(zero);

TEST(dyn_dispatch_FLOAT_T, zero)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(zero)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T out[TOTAL];

        // 先填充非零值
        for (size_t i = 0; i < TOTAL; ++i) out[i] = -1.0f;

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(zero)[idx](out);

        // 检查结果是否全为 0
        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], 0.0f);
    }
}
#endif

// ------------------------------------------ set ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void set(FLOAT_T x, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::set(x));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(set);

TEST(dyn_dispatch_FLOAT_T, set)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) out[i] = -1.0f;

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)[idx](3.5f, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], 3.5f);
    }
}
#endif

// ------------------------------------------ load + store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void load_store(const FLOAT_T* KSIMD_RESTRICT in, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::store(out + i, op::load(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(load_store);

TEST(dyn_dispatch_FLOAT_T, load_store)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = FLOAT_T(i + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = -1.0f;

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], in[i]);
    }
}
#endif

// ------------------------------------------ loadu + storeu ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void loadu_storeu(const FLOAT_T* KSIMD_RESTRICT in, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::loadu(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(loadu_storeu);

TEST(dyn_dispatch_FLOAT_T, loadu_storeu)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = FLOAT_T(i * 2 + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = -1.0f;

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], in[i]);
    }
}
#endif

// ------------------------------------------ add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void add(const FLOAT_T* KSIMD_RESTRICT a, const FLOAT_T* KSIMD_RESTRICT b, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::add(op::loadu(a + i), op::loadu(b + i)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(add);

TEST(dyn_dispatch_FLOAT_T, add)
{



    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(add)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL], b[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i);
            b[i] = FLOAT_T(100 + i);
            out[i] = -1.0f;
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(add)[idx](a, b, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], a[i] + b[i]);
    }
}
#endif

// ------------------------------------------ sub ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sub(const FLOAT_T* KSIMD_RESTRICT a, const FLOAT_T* KSIMD_RESTRICT b, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::sub(op::loadu(a + i), op::loadu(b + i)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(sub);

TEST(dyn_dispatch_FLOAT_T, sub)
{



    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sub)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL], b[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i*2 + 10);
            b[i] = FLOAT_T(i);
            out[i] = -1.0f;
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sub)[idx](a, b, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], a[i] - b[i]);
    }
}
#endif

// ------------------------------------------ mul ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul(const FLOAT_T* KSIMD_RESTRICT a, const FLOAT_T* KSIMD_RESTRICT b, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::mul(op::loadu(a + i), op::loadu(b + i)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(mul);

TEST(dyn_dispatch_FLOAT_T, mul)
{



    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mul)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL], b[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i+1);
            b[i] = FLOAT_T(2*(i+1));
            out[i] = -1.0f;
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mul)[idx](a, b, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], a[i] * b[i]);
    }
}
#endif

// ------------------------------------------ div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void div(const FLOAT_T* KSIMD_RESTRICT a, const FLOAT_T* KSIMD_RESTRICT b, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::div(op::loadu(a + i), op::loadu(b + i)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(div);

TEST(dyn_dispatch_FLOAT_T, div)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(div)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL], b[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T((i+1)*10);
            b[i] = FLOAT_T(i+1);
            out[i] = -1.0f;
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(div)[idx](a, b, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_FLOAT_EQ(out[i], a[i] / b[i]);
    }
}
#endif

// ------------------------------------------ one_div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void one_div(const FLOAT_T* KSIMD_RESTRICT a, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::one_div(op::loadu(a + i)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(one_div);

TEST(dyn_dispatch_FLOAT_T, one_div)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(one_div)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T((i+1)*10);
            out[i] = -1.0f;
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(one_div)[idx](a, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_NEAR(out[i], 1.0f / a[i], FLOAT_T_EPSILON_ONE_DIV);
    }
}
#endif

// ------------------------------------------ reduce_sum ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_sum(const FLOAT_T* KSIMD_RESTRICT in, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        FLOAT_T sum = 0.0f;
        for (size_t i = 0; i < TOTAL; i += Step)
        {
            sum += op::reduce_sum(op::loadu(in + i));
        }
        *out = sum;
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(reduce_sum);

TEST(dyn_dispatch_FLOAT_T, reduce_sum)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(reduce_sum)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T in[TOTAL];
        FLOAT_T out = 0.0f;

        for (size_t i = 0; i < TOTAL; ++i)
            in[i] = FLOAT_T(i+1); // 1,2,3,...

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(reduce_sum)[idx](in, &out);

        FLOAT_T expected = 0.0f;
        for (size_t i = 0; i < TOTAL; ++i)
            expected += in[i];

        EXPECT_FLOAT_EQ(out, expected);
    }
}
#endif

// ------------------------------------------ mul_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul_add(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        const FLOAT_T* KSIMD_RESTRICT c,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        FLOAT_T sum = 0.0f;
        for (size_t i = 0; i < TOTAL; i += Step)
        {
            sum += op::reduce_sum(op::mul_add(
                op::loadu(a + i),
                op::loadu(b + i),
                op::loadu(c + i)
            ));
        }
        *out = sum;
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(mul_add);

TEST(dyn_dispatch_FLOAT_T, mul_add)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mul_add)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        FLOAT_T r = 0.0f;

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i + 1);        // 1,2,3,...
            b[i] = FLOAT_T(i + 2);        // 2,3,4,...
            c[i] = FLOAT_T(i + 3);        // 3,4,5,...
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mul_add)[idx](a, b, c, &r);

        FLOAT_T expected = 0.0f;
        for (size_t i = 0; i < TOTAL; ++i)
            expected += a[i] * b[i] + c[i];

        EXPECT_FLOAT_EQ(r, expected);
    }
}
#endif

// ------------------------------------------ square ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void square(
        const FLOAT_T* KSIMD_RESTRICT a,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto r = op::square(op::load(a + i));
            op::store(out + i, r);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(square);

TEST(dyn_dispatch_FLOAT_T, square)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(square)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];

        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(square)[idx](a, b);
        EXPECT_TRUE(array_equal(b, std::size(b), 9));
    }
}
#endif

// ------------------------------------------ pow ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void pow(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto r = op::pow(op::load(a + i), op::load(b + i));
            op::store(out + i, r);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(pow);

TEST(dyn_dispatch_FLOAT_T, pow)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(pow)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 1.2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(pow)[idx](a, b, test);
        EXPECT_TRUE(array_approximately(test, std::size(test), std::pow(3.0, 1.2), FLOAT_T_EPSILON));
    }
}
#endif

// ------------------------------------------ sqrt ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sqrt(
        const FLOAT_T* KSIMD_RESTRICT a,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::sqrt(op::load(a + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(sqrt);

TEST(dyn_dispatch_FLOAT_T, sqrt)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sqrt)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i + 1);
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sqrt)[idx](a, r);

        for (size_t i = 0; i < TOTAL; ++i)
        {
            EXPECT_NEAR(r[i], std::sqrt(a[i]), FLOAT_T_EPSILON);
        }
    }
}
#endif

// ------------------------------------------ rsqrt ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void rsqrt(
        const FLOAT_T* KSIMD_RESTRICT a,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::rsqrt(op::load(a + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(rsqrt);

TEST(dyn_dispatch_FLOAT_T, rsqrt)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(rsqrt)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i)
        {
            a[i] = FLOAT_T(i + 1);
        }

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(rsqrt)[idx](a, r);

        for (size_t i = 0; i < TOTAL; ++i)
        {
            EXPECT_NEAR(r[i], 1.0 / std::sqrt(a[i]), FLOAT_T_EPSILON_RSQRT);
        }
    }
}
#endif

// ------------------------------------------ abs ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void abs(
        const FLOAT_T* KSIMD_RESTRICT a,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::abs(op::load(a + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(abs);

TEST(dyn_dispatch_FLOAT_T, abs)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(abs)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // positive test
        FILL_ARRAY(a, 2);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(abs)[idx](a, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // negative test
        FILL_ARRAY(a, -2);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(abs)[idx](a, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // +0
        FILL_ARRAY(a, +0.0);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(abs)[idx](a, r);
        EXPECT_TRUE(array_bit_equal(r, std::size(r), tsimd::zero_block<FLOAT_T>));

        // -0
        FILL_ARRAY(a, -0.0);
        // test sign bit is 0
        for (size_t i = 0; i < std::size(r); ++i)
        {
            constexpr auto bit_idx = tsimd::inverse_bit_index<FLOAT_T, 0>;
            EXPECT_FALSE(test_bit(r[i], tsimd::inverse_bit_index<FLOAT_T, bit_idx>));
        }
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(abs)[idx](a, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 0.0));
    }
}
#endif

// ------------------------------------------ min ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void min(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::min(op::load(a + i), op::load(b + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(min);

TEST(dyn_dispatch_FLOAT_T, min)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(min)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // min
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(min)[idx](a, b, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // equal
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(min)[idx](a, b, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // max
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(min)[idx](a, b, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
    }
}
#endif

// ------------------------------------------ max ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void max(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::max(op::load(a + i), op::load(b + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(max);

TEST(dyn_dispatch_FLOAT_T, max)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(max)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T r[TOTAL];

        // min
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(max)[idx](a, b, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // equal
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(max)[idx](a, b, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // max
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(max)[idx](a, b, r);
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
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::clamp(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, x);
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
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // <
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & min == max
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & unsafe
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // <=
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // <= & unsafe
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
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
        // > & unsafe
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
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
        // >= & unsafe
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
    }
}
#endif

// ------------------------------------------ unsafe_clamp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void unsafe_clamp(
        const FLOAT_T* KSIMD_RESTRICT a,
        const FLOAT_T* KSIMD_RESTRICT b,
        const FLOAT_T* KSIMD_RESTRICT c,
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::unsafe_clamp(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, x);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(unsafe_clamp);

TEST(dyn_dispatch_FLOAT_T, unsafe_clamp)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)); ++idx)
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
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & min == max
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 1);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // in & unsafe
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 0);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_FALSE(array_equal(r, std::size(r), 1));

        // <
        FILL_ARRAY(a, 0);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));
        // < & min == max
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // <=
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 1));

        // >
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // > & min == max
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));

        // >=
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(c, 2);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(unsafe_clamp)[idx](a, b, c, r);
        EXPECT_TRUE(array_equal(r, std::size(r), 2));
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
        FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            using batch_t = traits::batch_t;

            batch_t x = op::lerp(op::load(a + i), op::load(b + i), op::load(t + i));
            op::store(out + i, x);
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
        FILL_ARRAY(t, 0.6);
        FILL_ARRAY(r, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lerp)[idx](a, b, t, r);
        EXPECT_TRUE(array_approximately(r, std::size(r), 1.2, FLOAT_T_EPSILON));

        // in & inv a b
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 0);
        FILL_ARRAY(t, 0.6);
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

// ------------------------------------------ equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(equal);

TEST(dyn_dispatch_FLOAT_T, equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 1
        FILL_ARRAY(a, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 3
        FILL_ARRAY(a, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(b, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_equal);

TEST(dyn_dispatch_FLOAT_T, not_equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN one test 1
        FILL_ARRAY(a, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 3
        FILL_ARRAY(a, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(b, std::numeric_limits<FLOAT_T>::quiet_NaN());
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ greater ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void greater(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::greater(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(greater);

TEST(dyn_dispatch_FLOAT_T, greater)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_greater ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_greater(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_greater(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_greater);

TEST(dyn_dispatch_FLOAT_T, not_greater)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN one test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ greater_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void greater_equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::greater_equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(greater_equal);

TEST(dyn_dispatch_FLOAT_T, greater_equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_greater_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_greater_equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_greater_equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_greater_equal);

TEST(dyn_dispatch_FLOAT_T, not_greater_equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 1);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN one test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_greater_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ less ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void less(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::less(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(less);

TEST(dyn_dispatch_FLOAT_T, less)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 5);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_less ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_less(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {


        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_less(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_less);

TEST(dyn_dispatch_FLOAT_T, not_less)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN one test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ less_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void less_equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::less_equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(less_equal);

TEST(dyn_dispatch_FLOAT_T, less_equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 5);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN zero test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_less_equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_less_equal(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_less_equal(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_less_equal);

TEST(dyn_dispatch_FLOAT_T, not_less_equal)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 2);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // NaN one test 1
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 2
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // NaN one test 3
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_less_equal)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ any_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void any_NaN(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::any_NaN(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(any_NaN);

TEST(dyn_dispatch_FLOAT_T, any_NaN)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(any_NaN)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // zero test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(any_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(any_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(any_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // one test
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(any_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ not_NaN ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void not_NaN(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::not_NaN(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(not_NaN);

TEST(dyn_dispatch_FLOAT_T, not_NaN)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_NaN)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        // one test
        FILL_ARRAY(a, 3);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::one_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, 2);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, 1);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));

        // zero test
        FILL_ARRAY(a, qNaN<FLOAT_T>);
        FILL_ARRAY(b, qNaN<FLOAT_T>);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(not_NaN)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), tsimd::zero_block<FLOAT_T>));
    }
}
#endif

// ------------------------------------------ bit_not ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_not(
        const FLOAT_T* a,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_not(op::load(a + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_not);

TEST(dyn_dispatch_FLOAT_T, bit_not)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_not)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_not)[idx](a, test);

        // 只判断前5个bit是否测试成功
        // expected: 01010
        for (size_t i = 0; i < std::size(test); ++i)
        {
            auto result_bits = tsimd::detail::bitcast_to_uint(test[i]);

            EXPECT_FALSE(test_bit(result_bits, 0));
            EXPECT_TRUE(test_bit(result_bits, 1));
            EXPECT_FALSE(test_bit(result_bits, 2));
            EXPECT_TRUE(test_bit(result_bits, 3));
            EXPECT_FALSE(test_bit(result_bits, 4));
        }
    }
}
#endif

// ------------------------------------------ bit_and ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_and(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_and(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_and);

TEST(dyn_dispatch_FLOAT_T, bit_and)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_and)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<FLOAT_T>(0b10011));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_and)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<FLOAT_T>(0b10001)));
    }
}
#endif

// ------------------------------------------ bit_and_not ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_and_not(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_and_not(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_and_not);

TEST(dyn_dispatch_FLOAT_T, bit_and_not)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_and_not)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<FLOAT_T>(0b10011));
        FILL_ARRAY(test, -1);
        // not: a = 01010
        //      b = 10011
        // and: r = 00010
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_and_not)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<FLOAT_T>(0b00010)));
    }
}
#endif

// ------------------------------------------ bit_or ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_or(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_or(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_or);

TEST(dyn_dispatch_FLOAT_T, bit_or)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_or)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<FLOAT_T>(0b10011));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_or)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<FLOAT_T>(0b10111)));
    }
}
#endif

// ------------------------------------------ bit_xor ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_xor(
        const FLOAT_T* a,
        const FLOAT_T* b,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_xor(op::load(a + i), op::load(b + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_xor);

TEST(dyn_dispatch_FLOAT_T, bit_xor)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_xor)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<FLOAT_T>(0b10011));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_xor)[idx](a, b, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<FLOAT_T>(0b00110)));
    }
}
#endif

// ------------------------------------------ bit_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_select(
        const FLOAT_T* a,
        const FLOAT_T* b,
        const FLOAT_T* c,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_select(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_select);

TEST(dyn_dispatch_FLOAT_T, bit_select)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_select)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<FLOAT_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<FLOAT_T>(0b11111));
        FILL_ARRAY(c, make_float_from_bits<FLOAT_T>(0b00010));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_select)[idx](a, b, c, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<FLOAT_T>(0b10111)));
    }
}
#endif

// ------------------------------------------ sign_bit_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sign_bit_select(
        const FLOAT_T* a,
        const FLOAT_T* b,
        const FLOAT_T* c,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::sign_bit_select(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(sign_bit_select);

TEST(dyn_dispatch_FLOAT_T, sign_bit_select)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sign_bit_select)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, 0);
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            a[i] = tsimd::sign_bit_mask<FLOAT_T>; // 0和偶数变成 sign bit 1
        }
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 3);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sign_bit_select)[idx](a, b, c, test);
        // 偶数是b
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == b[i]);
        }
        // 奇数是c
        for (size_t i = 1; i + 1 < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == c[i]);
        }
    }
}
#endif

// ------------------------------------------ lane_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void lane_select(
        const FLOAT_T* a,
        const FLOAT_T* b,
        const FLOAT_T* c,
        FLOAT_T* out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_TRAITS(FLOAT_T);
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::lane_select(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(lane_select);

TEST(dyn_dispatch_FLOAT_T, lane_select)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lane_select)); ++idx)
    {
        alignas(ALIGNMENT) FLOAT_T a[TOTAL];
        alignas(ALIGNMENT) FLOAT_T b[TOTAL];
        alignas(ALIGNMENT) FLOAT_T c[TOTAL];
        alignas(ALIGNMENT) FLOAT_T test[TOTAL];

        FILL_ARRAY(a, 0);
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            a[i] = 1; // 0和偶数变成1
        }
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 3);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lane_select)[idx](a, b, c, test);
        // 偶数是b
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == b[i]);
        }
        // 奇数是c
        for (size_t i = 1; i + 1 < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == c[i]);
        }

        // NaN test
        FILL_ARRAY(a, 0);
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            a[i] = qNaN<FLOAT_T>; // 0和偶数变成NaN
        }
        FILL_ARRAY(b, 2);
        FILL_ARRAY(c, 3);
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(lane_select)[idx](a, b, c, test);
        // 偶数是b
        for (size_t i = 0; i < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == b[i]);
        }
        // 奇数是c
        for (size_t i = 1; i + 1 < TOTAL; i += 2)
        {
            EXPECT_TRUE(test[i] == c[i]);
        }
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
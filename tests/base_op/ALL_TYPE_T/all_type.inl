// using ALL_TYPE_T = uint32_t;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>

KSIMD_WARNING_PUSH
KSIMD_IGNORE_WARNING_MSVC(4723) // ignore warning: divide by 0

// ------------------------------------------ undefined ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void undefined() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;

        [[maybe_unused]] batch_t z = op::undefined();
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(undefined)
#endif

// ------------------------------------------ zero ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void zero() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = op::batch_t;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T arr[Lanes]{};
        std::memset(arr, 0xff, sizeof(arr));

        batch_t z = op::zero();
        op::store(arr, z);
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(arr[i] == TYPE_T(0));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(zero)
#endif


// ------------------------------------------ set ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void set() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T arr[Lanes];

        // 测试常规数值广播
        TYPE_T val = TYPE_T(42);
        batch_t v = op::set(val);
        op::store(arr, v);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(arr[i], val);

        // 针对浮点数的特殊值测试
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN 广播
            op::store(arr, op::set(qNaN<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(arr[i]));

            // Inf 广播
            op::store(arr, op::set(inf<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(arr[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(set)
#endif

// ------------------------------------------ sequence ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sequence() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T arr[Lanes];

        // 1. 无参 sequence(): [0, 1, 2, ...]
        op::store(arr, op::sequence());
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(i));
        }

        // 2. 带 base: [base, base + 1, ...]
        TYPE_T base = TYPE_T(10);
        op::store(arr, op::sequence(base));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(base + static_cast<TYPE_T>(i)));
        }

        // 3. 带 base 和 stride: [base, base + stride, ...]
        TYPE_T b_v = TYPE_T(5), stride = TYPE_T(2);
        op::store(arr, op::sequence(b_v, stride));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(b_v + static_cast<TYPE_T>(i) * stride));
        }

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            TYPE_T f_base = TYPE_T(1.5), f_stride = TYPE_T(-0.5);
            op::store(arr, op::sequence(f_base, f_stride));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_NEAR(arr[i], f_base + static_cast<TYPE_T>(i) * f_stride, TYPE_T(1e-6));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(sequence)
#endif

// ------------------------------------------ load_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void load_store() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T in[Lanes];
        alignas(op::Alignment) TYPE_T out[Lanes];

        for (size_t i = 0; i < Lanes; ++i) {
            in[i] = TYPE_T(i + 7);
            out[i] = TYPE_T(0);
        }

        op::store(out, op::load(in));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(out[i], in[i]);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(load_store)
#endif

// ------------------------------------------ loadu_storeu ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void loadu_storeu() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        // 分配略大空间以模拟非对齐
        alignas(op::Alignment) TYPE_T buffer_in[Lanes + 1];
        alignas(op::Alignment) TYPE_T buffer_out[Lanes + 1];

        TYPE_T* u_in = buffer_in + 1;
        TYPE_T* u_out = buffer_out + 1;

        for (size_t i = 0; i < Lanes; ++i) {
            u_in[i] = TYPE_T(i * 3 + 1);
            u_out[i] = TYPE_T(0);
        }

        op::storeu(u_out, op::loadu(u_in));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(u_out[i], u_in[i]);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(loadu_storeu)
#endif

// ------------------------------------------ partial_load_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void partial_load_store() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T in[Lanes * 2];
        alignas(op::Alignment) TYPE_T out[Lanes];

        for (size_t i = 0; i < Lanes * 2; ++i) in[i] = TYPE_T(i + 1);

        // 1. load_partial & zero-padding check
        for (size_t n = 0; n <= Lanes; ++n) {
            std::memset(out, 0xAA, sizeof(out)); // 干扰值
            batch_t v = op::load_partial(in, n);
            op::store(out, v);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], in[i]);
                else EXPECT_EQ(out[i], TYPE_T(0)); // 必须清零
            }
        }

        // 2. store_partial & memory protection
        for (size_t n = 0; n <= Lanes; ++n) {
            constexpr TYPE_T sentinel = TYPE_T(88);
            for (size_t i = 0; i < Lanes; ++i) out[i] = sentinel;

            batch_t v = op::set(TYPE_T(99));
            op::store_partial(out, v, n);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], TYPE_T(99));
                else EXPECT_EQ(out[i], sentinel); // 不应触碰
            }
        }

        // 3. Unaligned safety
        if constexpr (Lanes > 1) {
            batch_t v = op::load_partial(in + 1, 1);
            op::store(out, v);
            EXPECT_EQ(out[0], in[1]);
            EXPECT_EQ(out[1], TYPE_T(0));
        }

        // 4. Overflow tolerance (n > Lanes)
{
    batch_t v = op::load_partial(in, Lanes + 10);
    op::store(out, v);
    EXPECT_EQ(out[Lanes - 1], in[Lanes - 1]);
}
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(partial_load_store)
#endif

// ------------------------------------------ bit_if_then_else ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_if_then_else() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T res[Lanes];

        // 测试数据：验证位选择逻辑 (mask & a) | (~mask & b)
        TYPE_T val_a    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b10101));
        TYPE_T val_b    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b11111));
        TYPE_T val_mask = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b00010));
        uint_t expected = static_cast<uint_t>(0b11101);

        op::store(res, op::bit_if_then_else(op::set(val_mask), op::set(val_a), op::set(val_b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(expected)))
                << "Bit select failed at lane " << i
                << "\n  Expected bits: 0x" << std::hex << (uint64_t)expected
                << "\n  Actual bits:   0x" << (uint64_t)std::bit_cast<uint_t>(res[i]);
        }

        // 浮点数特殊验证：符号位搬运
        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            TYPE_T pos_val = TYPE_T(1.0);
            TYPE_T neg_val = TYPE_T(-2.0);
            TYPE_T s_mask  = ksimd::SignBitMask<TYPE_T>;

            // 从 neg_val 取符号位，从 pos_val 取数值位，结果应为 -1.0
            op::store(res, op::bit_if_then_else(op::set(s_mask), op::set(neg_val), op::set(pos_val)));

            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(res[i], TYPE_T(-1.0));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_if_then_else)
#endif

// ------------------------------------------ if_then_else ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void if_then_else() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T res[Lanes];

        batch_t v_a = op::set(TYPE_T(10));
        batch_t v_b = op::set(TYPE_T(20));

        // 1. 全 1 掩码选择
        {
            auto mask_true = op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(1)));
            op::store(res, op::if_then_else(mask_true, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(10));
        }

        // 2. 全 0 掩码选择
        {
            auto mask_false = op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(2)));
            op::store(res, op::if_then_else(mask_false, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(20));
        }

        // 3. 混合掩码交叉选择
        {
            alignas(op::Alignment) TYPE_T data_lhs[Lanes];
            alignas(op::Alignment) TYPE_T data_rhs[Lanes];
            for (size_t i = 0; i < Lanes; ++i) {
                data_lhs[i] = static_cast<TYPE_T>(i);
                data_rhs[i] = TYPE_T(1);
            }

            auto mask_mixed = op::greater(op::load(data_lhs), op::load(data_rhs));
            op::store(res, op::if_then_else(mask_mixed, v_a, v_b));

            for (size_t i = 0; i < Lanes; ++i) {
                TYPE_T expected = (i > 1) ? TYPE_T(10) : TYPE_T(20);
                EXPECT_EQ(res[i], expected) << "Lane " << i << " failed";
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(if_then_else)
#endif

// ------------------------------------------ bit_not ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_not() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;

        alignas(op::Alignment) TYPE_T res[Lanes];

        // 输入数据: ...010101 (0x15) -> 取反期望: ...101010 (低5位)
        uint_t input_bits = 0b10101;
        TYPE_T input_val = make_var_from_bits<TYPE_T>(static_cast<uint_t>(input_bits));

        op::store(res, op::bit_not(op::set(input_val)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            // 验证低5位翻转
            EXPECT_FALSE(test_bit(res[i], 0)); // 1 -> 0
            EXPECT_TRUE(test_bit(res[i], 1));  // 0 -> 1
            EXPECT_FALSE(test_bit(res[i], 2)); // 1 -> 0
            EXPECT_TRUE(test_bit(res[i], 3));  // 0 -> 1
            EXPECT_FALSE(test_bit(res[i], 4)); // 1 -> 0
        }

        // 全 0 取反验证
        {
            uint_t zero_bits = 0;
            uint_t expected_bits = ~zero_bits;
            op::store(res, op::bit_not(op::set(make_var_from_bits<TYPE_T>(zero_bits))));

            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(expected_bits)));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_not)
#endif

// ------------------------------------------ bit_and ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_and() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;

        alignas(op::Alignment) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10001
        uint_t a = 0b10101, b = 0b10011, exp = 0b10001;

        op::store(res, op::bit_and(op::set(make_var_from_bits<TYPE_T>(a)),
                                   op::set(make_var_from_bits<TYPE_T>(b))));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(exp)));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_and)
#endif

// ------------------------------------------ bit_and_not ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_and_not() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;

        alignas(op::Alignment) TYPE_T res[Lanes];

        // 逻辑通常为: (~a) & b
        // a: 10101 (~a 低位: 01010)
        // b: 10011
        // res: 00010
        uint_t a = 0b10101, b = 0b10011, exp = 0b00010;

        op::store(res, op::bit_and_not(op::set(make_var_from_bits<TYPE_T>(a)),
                                       op::set(make_var_from_bits<TYPE_T>(b))));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(exp)));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_and_not)
#endif

// ------------------------------------------ bit_or ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_or() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;

        alignas(op::Alignment) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10111
        uint_t a = 0b10101, b = 0b10011, exp = 0b10111;

        op::store(res, op::bit_or(op::set(make_var_from_bits<TYPE_T>(a)),
                                  op::set(make_var_from_bits<TYPE_T>(b))));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(exp)));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_or)
#endif

// ------------------------------------------ bit_xor ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_xor() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;

        alignas(op::Alignment) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 00110
        uint_t a = 0b10101, b = 0b10011, exp = 0b00110;

        op::store(res, op::bit_xor(op::set(make_var_from_bits<TYPE_T>(a)),
                                   op::set(make_var_from_bits<TYPE_T>(b))));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(exp)));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(bit_xor)
#endif

// ------------------------------------------ add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 常规数值测试
        op::store(test, op::add(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        // 浮点特殊边界测试
        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            // Inf + 1 = Inf
            op::store(test, op::add(op::set(inf<TYPE_T>), op::set(TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

            // NaN + 1 = NaN
            op::store(test, op::add(op::set(qNaN<TYPE_T>), op::set(TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf + (-Inf) = NaN
            op::store(test, op::add(op::set(inf<TYPE_T>), op::set(-inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(add)
#endif

// ------------------------------------------ sub ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sub() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 常规数值测试
        op::store(test, op::sub(op::set(TYPE_T(50)), op::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            // Inf - Inf = NaN
            op::store(test, op::sub(op::set(inf<TYPE_T>), op::set(inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // 1.0 - NaN = NaN
            op::store(test, op::sub(op::set(TYPE_T(1)), op::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(sub)
#endif

// ------------------------------------------ mul ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 常规数值测试
        op::store(test, op::mul(op::set(TYPE_T(6)), op::set(TYPE_T(7))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(42));

        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            // Inf * 0 = NaN
            op::store(test, op::mul(op::set(inf<TYPE_T>), op::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf * (-2) = -Inf
            op::store(test, op::mul(op::set(inf<TYPE_T>), op::set(TYPE_T(-2))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] < 0);
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mul)
#endif

// ------------------------------------------ div ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void div() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 常规数值测试
        op::store(test, op::div(op::set(TYPE_T(100)), op::set(TYPE_T(4))));
        for (size_t i = 0; i < Lanes; ++i) {
            // 使用标准 EXPECT_NEAR 验证除法精度
            EXPECT_NEAR(static_cast<double>(test[i]), 25.0, 1e-7);
        }

        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            // 1.0 / 0.0 = Inf
            op::store(test, op::div(op::set(TYPE_T(1)), op::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

            // 0.0 / 0.0 = NaN
            op::store(test, op::div(op::set(TYPE_T(0)), op::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf / Inf = NaN
            op::store(test, op::div(op::set(inf<TYPE_T>), op::set(inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(div)
#endif

// ------------------------------------------ reduce_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T data[Lanes];
        TYPE_T expected = 0;
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
            expected += data[i];
        }

        TYPE_T res = op::reduce_add(op::load(data));
        EXPECT_NEAR((res), (expected), std::numeric_limits<TYPE_T>::epsilon() * 10);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Inf in sum
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(op::reduce_add(op::load(data))));

            // NaN in sum
            data[0] = qNaN<TYPE_T>;
            EXPECT_TRUE(std::isnan(op::reduce_add(op::load(data))));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_add)
#endif

// ------------------------------------------ mul_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul_add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        // using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // (2 * 3) + 4 = 10
        op::store(test, op::mul_add(op::set(TYPE_T(2)), op::set(TYPE_T(3)), op::set(TYPE_T(4))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN propagation
            op::store(test, op::mul_add(op::set(qNaN<TYPE_T>), op::set(TYPE_T(2)), op::set(TYPE_T(3))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf propagation
            op::store(test, op::mul_add(op::set(inf<TYPE_T>), op::set(TYPE_T(2)), op::set(TYPE_T(3))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mul_add)
#endif

// ------------------------------------------ min ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void min() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        op::store(test, op::min(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Min(Inf, 100) = 100
            op::store(test, op::min(op::set(inf<TYPE_T>), op::set(TYPE_T(100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // Min(100, Inf) = 100
            op::store(test, op::min(op::set(TYPE_T(100)), op::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // NaN 行为 (依照指令集惯例，通常返回非 NaN 操作数，或者取决于位置)
            op::store(test, op::min(op::set(qNaN<TYPE_T>), op::set(TYPE_T(5))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(5)));

            op::store(test, op::min(op::set(TYPE_T(5)), op::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void max() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        op::store(test, op::max(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(20)));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Max(-Inf, -100) = -100
            op::store(test, op::max(op::set(-inf<TYPE_T>), op::set(TYPE_T(-100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));

            // NaN 行为
            op::store(test, op::max(op::set(qNaN<TYPE_T>), op::set(TYPE_T(-100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(min)
TEST_ONCE_DYN(max)
#endif

// ------------------------------------------ equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 == 1 (True)
        op::test_store_mask(test, op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        // 1 == 2 (False)
        op::test_store_mask(test, op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN == NaN (False)
            op::test_store_mask(test, op::equal(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void not_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 != 2 (True)
        op::test_store_mask(test, op::not_equal(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN != NaN (True)
            op::test_store_mask(test, op::not_equal(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(equal)
TEST_ONCE_DYN(not_equal)
#endif

// ------------------------------------------ greater ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void greater() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 2 > 1 (True), 1 > 2 (False)
        op::test_store_mask(test, op::greater(op::set(TYPE_T(2)), op::set(TYPE_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Inf > 1e30 (True)
            op::test_store_mask(test, op::greater(op::set(inf<TYPE_T>), op::set(TYPE_T(1e30))));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // 1 > -Inf (True)
            op::test_store_mask(test, op::greater(op::set(TYPE_T(1)), op::set(-inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN > 任何数 (False)
            op::test_store_mask(test, op::greater(op::set(qNaN<TYPE_T>), op::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void greater_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 2 >= 2 (True)
        op::test_store_mask(test, op::greater_equal(op::set(TYPE_T(2)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(greater)
TEST_ONCE_DYN(greater_equal)
#endif

// ------------------------------------------ less ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void less() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 < 2 (True)
        op::test_store_mask(test, op::less(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // -Inf < Inf (True)
            op::test_store_mask(test, op::less(op::set(-inf<TYPE_T>), op::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN 相关比较应为 False
            op::test_store_mask(test, op::less(op::set(-inf<TYPE_T>), op::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

            op::test_store_mask(test, op::less(op::set(qNaN<TYPE_T>), op::set(-inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void less_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 5 <= 5 (True)
        op::test_store_mask(test, op::less_equal(op::set(TYPE_T(5)), op::set(TYPE_T(5))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(less)
TEST_ONCE_DYN(less_equal)
#endif

// ------------------------------------------ mask_logic ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_logic() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;
        using mask_t = ns::Mask<TYPE_T>;

        // 准备测试数据
        batch_t v1 = op::set(static_cast<TYPE_T>(10));
        batch_t v2 = op::set(static_cast<TYPE_T>(20));

        mask_t m_true  = op::equal(v1, v1);  // All Ones
        mask_t m_false = op::equal(v1, v2);  // All Zeros

        #define KSIMD_CHECK_MASK_EQ(lhs, rhs) \
        do { \
            alignas(op::Alignment) TYPE_T M__l[op::Lanes]{}; \
            alignas(op::Alignment) TYPE_T M__r[op::Lanes]{}; \
            op::test_store_mask(M__l, lhs); \
            op::test_store_mask(M__r, rhs); \
            for (size_t I__ = 0; I__ < op::Lanes; ++I__) \
            { \
                EXPECT_TRUE(bit_equal(M__l[I__], M__r[I__])); \
            } \
        } while (0)

        // 1. 基础位运算函数测试 (and, or, xor, not)
        {
            KSIMD_CHECK_MASK_EQ(op::mask_and(m_true, m_false), m_false);
            KSIMD_CHECK_MASK_EQ(op::mask_or(m_true, m_false),  m_true);
            KSIMD_CHECK_MASK_EQ(op::mask_xor(m_true, m_true),  m_false);
            KSIMD_CHECK_MASK_EQ(op::mask_not(m_true),          m_false);
            KSIMD_CHECK_MASK_EQ(op::mask_not(m_false),         m_true);
        }

        // 2. 运算符重载测试 (operator &, |, ^, ~)
        {
            KSIMD_CHECK_MASK_EQ(m_true & m_false, m_false);
            KSIMD_CHECK_MASK_EQ(m_true | m_false, m_true);
            KSIMD_CHECK_MASK_EQ(m_true ^ m_true,  m_false);
            KSIMD_CHECK_MASK_EQ(~m_true,          m_false);
        }

        // 3. 复合赋值运算符测试 (operator &=, |=, ^=)
        {
            mask_t m_curr = m_true;

            m_curr &= m_false;
            KSIMD_CHECK_MASK_EQ(m_curr, m_false);

            m_curr |= m_true;
            KSIMD_CHECK_MASK_EQ(m_curr, m_true);

            m_curr ^= m_true;
            KSIMD_CHECK_MASK_EQ(m_curr, m_false);
        }

        #undef KSIMD_CHECK_MASK_EQ
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mask_logic)
#endif

// ------------------------------------------ mask_operator_overload ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_operator_overload() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;
        using mask_t = ns::Mask<TYPE_T>;

        // 准备基础掩码
        batch_t v1 = op::set(static_cast<TYPE_T>(10));
        batch_t v2 = op::set(static_cast<TYPE_T>(20));
        mask_t m_true  = op::equal(v1, v1); // 全1
        mask_t m_false = op::equal(v1, v2); // 全0

        #define KSIMD_CHECK_MASK_EQ(lhs, rhs) \
        do { \
            alignas(op::Alignment) TYPE_T M__l[op::Lanes]{}; \
            alignas(op::Alignment) TYPE_T M__r[op::Lanes]{}; \
            op::test_store_mask(M__l, lhs); \
            op::test_store_mask(M__r, rhs); \
            for (size_t I__ = 0; I__ < op::Lanes; ++I__) \
            { \
                EXPECT_TRUE(bit_equal(M__l[I__], M__r[I__])); \
            } \
        } while (0)

        // 1. 测试一元运算符: ~ (NOT)
        {
            KSIMD_CHECK_MASK_EQ(~m_true,  m_false);
            KSIMD_CHECK_MASK_EQ(~m_false, m_true);
        }

        // 2. 测试二元运算符: &, |, ^ (AND, OR, XOR)
        {
            KSIMD_CHECK_MASK_EQ(m_true  & m_false, m_false);
            KSIMD_CHECK_MASK_EQ(m_true  | m_false, m_true);
            KSIMD_CHECK_MASK_EQ(m_true  ^ m_true,  m_false);
            KSIMD_CHECK_MASK_EQ(m_true  ^ m_false, m_true);
        }

        // 3. 测试复合赋值运算符: &=, |=, ^=
        {
            mask_t m = m_true;

            m &= m_false;
            KSIMD_CHECK_MASK_EQ(m, m_false);

            m |= m_true;
            KSIMD_CHECK_MASK_EQ(m, m_true);

            m ^= m_true; // True ^ True = False
            KSIMD_CHECK_MASK_EQ(m, m_false);
        }

        #undef KSIMD_CHECK_MASK_EQ
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mask_operator_overload)
#endif

// ------------------------------------------ all_operators ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void all_operators() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        using batch_t = ns::Batch<TYPE_T>;

        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T act[Lanes];
        alignas(op::Alignment) TYPE_T exp[Lanes];

        // 辅助检查函数：对比两个 Batch 的存储内容
        #define check_eq(lhs, rhs, label) \
            do { \
                op::store(act, lhs); \
                op::store(exp, rhs); \
                for (size_t i = 0; i < Lanes; ++i) { \
                    EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Operator " << label << " forwarding failed at lane " << i; \
                } \
            } while (0)

        // 构造简单的测试数据
        batch_t a = op::set(TYPE_T(10));
        batch_t b = op::set(TYPE_T(2));

        // 1. 二元算术转发验证
        check_eq(a + b, op::add(a, b), "+");
        check_eq(a - b, op::sub(a, b), "-");
        check_eq(a * b, op::mul(a, b), "*");
        check_eq(a / b, op::div(a, b), "/");

        // 2. 位运算转发验证
        check_eq(a & b, op::bit_and(a, b), "&");
        check_eq(a | b, op::bit_or(a, b), "|");
        check_eq(a ^ b, op::bit_xor(a, b), "^");
        check_eq(~a,    op::bit_not(a), "~");

        // 3. 一元负号转发验证
        check_eq(-a, op::sub(op::set(TYPE_T(0)), a), "unary -");

        // 4. 复合赋值转发验证
        batch_t c = a;
        check_eq(c += b, op::add(a, b), "+=");

        c = a;
        check_eq(c -= b, op::sub(a, b), "-=");

        c = a;
        check_eq(c &= b, op::bit_and(a, b), "&=");

        c = a;
        check_eq(c ^= b, op::bit_xor(a, b), "^=");
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(all_operators)
#endif

// ------------------------------------------ Comparison Operators ------------------------------------------

namespace KSIMD_DYN_INSTRUCTION
{
    // --- Helper for Comparison Tests ---
    #define KSIMD_CHECK_COMP_OP(batch_lhs, batch_rhs, op_symbol, expected_block) \
    do { \
        op::test_store_mask(test, (batch_lhs) op_symbol (batch_rhs)); \
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::expected_block<TYPE_T>)) \
            << "Comparison failed for: " << #batch_lhs << " " << #op_symbol << " " << #batch_rhs; \
    } while (0)

    KSIMD_DYN_FUNC_ATTR
    void op_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 == 1 (True), 1 == 2 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), ==, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), ==, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(inf<TYPE_T>), op::set(inf<TYPE_T>), ==, OneBlock);
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>), ==, ZeroBlock); // NaN != NaN
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void op_not_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 != 2 (True), 1 != 1 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), !=, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), !=, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>), !=, OneBlock); // NaN != NaN is True
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void op_greater() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 2 > 1 (True), 1 > 2 (False), 1 > 1 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(2)), op::set(TYPE_T(1)), >, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), >, ZeroBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), >, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(inf<TYPE_T>), op::set(-inf<TYPE_T>), >, OneBlock);
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(TYPE_T(0)), >, ZeroBlock);
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void op_greater_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 2 >= 1 (True), 1 >= 1 (True), 1 >= 2 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(2)), op::set(TYPE_T(1)), >=, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), >=, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), >=, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(inf<TYPE_T>), op::set(inf<TYPE_T>), >=, OneBlock);
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>), >=, ZeroBlock);
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void op_less() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 < 2 (True), 2 < 1 (False), 1 < 1 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), <, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(2)), op::set(TYPE_T(1)), <, ZeroBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), <, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(-inf<TYPE_T>), op::set(inf<TYPE_T>), <, OneBlock);
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(inf<TYPE_T>), <, ZeroBlock);
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void op_less_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) TYPE_T test[Lanes];

        // 1 <= 2 (True), 1 <= 1 (True), 2 <= 1 (False)
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(2)), <=, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(1)), op::set(TYPE_T(1)), <=, OneBlock);
        KSIMD_CHECK_COMP_OP(op::set(TYPE_T(2)), op::set(TYPE_T(1)), <=, ZeroBlock);

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            KSIMD_CHECK_COMP_OP(op::set(-inf<TYPE_T>), op::set(-inf<TYPE_T>), <=, OneBlock);
            KSIMD_CHECK_COMP_OP(op::set(qNaN<TYPE_T>), op::set(TYPE_T(0)), <=, ZeroBlock);
        }
    }

    #undef KSIMD_CHECK_COMP_OP
}

#if KSIMD_ONCE
TEST_ONCE_DYN(op_equal)
TEST_ONCE_DYN(op_not_equal)
TEST_ONCE_DYN(op_greater)
TEST_ONCE_DYN(op_greater_equal)
TEST_ONCE_DYN(op_less)
TEST_ONCE_DYN(op_less_equal)
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif

KSIMD_WARNING_POP

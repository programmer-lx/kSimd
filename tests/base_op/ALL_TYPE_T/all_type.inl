// using ALL_TYPE_T = uint32_t;

#if _WIN32 || WIN64
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
    #include <windows.h>
#endif

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

        [[maybe_unused]] ns::Batch<TYPE_T> z = ns::undefined<TYPE_T>();
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

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T arr[Lanes]{};
        std::memset(arr, 0xff, sizeof(arr));

        ns::Batch<TYPE_T> z = ns::zero<TYPE_T>();
        ns::store(arr, z);
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

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T arr[Lanes];

        // 测试常规数值广播
        TYPE_T val = TYPE_T(42);
        ns::Batch<TYPE_T> v = ns::set(val);
        ns::store(arr, v);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(arr[i], val);

        // 针对浮点数的特殊值测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN 广播
            ns::store(arr, ns::set(qNaN<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(arr[i]));

            // Inf 广播
            ns::store(arr, ns::set(inf<TYPE_T>));
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
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T arr[Lanes];

        // 1. 无参 sequence(): [0, 1, 2, ...]
        ns::store(arr, ns::sequence<TYPE_T>());
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(i));
        }

        // 2. 带 base: [base, base + 1, ...]
        TYPE_T base = TYPE_T(10);
        ns::store(arr, ns::sequence<TYPE_T>(base));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(base + static_cast<TYPE_T>(i)));
        }

        // 3. 带 base 和 stride: [base, base + stride, ...]
        TYPE_T b_v = TYPE_T(5), stride = TYPE_T(2);
        ns::store(arr, ns::sequence<TYPE_T>(b_v, stride));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(b_v + static_cast<TYPE_T>(i) * stride));
        }

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            TYPE_T f_base = TYPE_T(1.5), f_stride = TYPE_T(-0.5);
            ns::store(arr, ns::sequence<TYPE_T>(f_base, f_stride));
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
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T in[Lanes];
        alignas(ns::Alignment<TYPE_T>) TYPE_T out[Lanes];

        for (size_t i = 0; i < Lanes; ++i) {
            in[i] = TYPE_T(i + 7);
            out[i] = TYPE_T(0);
        }

        ns::store(out, ns::load(in));
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
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        // 分配略大空间以模拟非对齐
        alignas(ns::Alignment<TYPE_T>) TYPE_T buffer_in[Lanes + 1];
        alignas(ns::Alignment<TYPE_T>) TYPE_T buffer_out[Lanes + 1];

        TYPE_T* u_in = buffer_in + 1;
        TYPE_T* u_out = buffer_out + 1;

        for (size_t i = 0; i < Lanes; ++i) {
            u_in[i] = TYPE_T(i * 3 + 1);
            u_out[i] = TYPE_T(0);
        }

        ns::storeu(u_out, ns::loadu(u_in));
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T in[Lanes * 2];
        alignas(ns::Alignment<TYPE_T>) TYPE_T out[Lanes];

        for (size_t i = 0; i < Lanes * 2; ++i) in[i] = TYPE_T(i + 1);

        // 1. loadu_partial & zero-padding check
        for (size_t n = 0; n <= Lanes; ++n) {
            std::memset(out, 0xAA, sizeof(out)); // 干扰值
            ns::Batch<TYPE_T> v = ns::loadu_partial(in, n);
            ns::store(out, v);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], in[i]);
                else EXPECT_EQ(out[i], TYPE_T(0)); // 必须清零
            }
        }

        // 2. storeu_partial & memory protection
        for (size_t n = 0; n <= Lanes; ++n) {
            constexpr TYPE_T sentinel = TYPE_T(88);
            for (size_t i = 0; i < Lanes; ++i) out[i] = sentinel;

            ns::Batch<TYPE_T> v = ns::set(TYPE_T(99));
            ns::storeu_partial(out, v, n);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], TYPE_T(99));
                else EXPECT_EQ(out[i], sentinel); // 不应触碰
            }
        }

        // 3. Unaligned safety
        if constexpr (Lanes > 1) {
            ns::Batch<TYPE_T> v = ns::loadu_partial(in + 1, 1);
            ns::store(out, v);
            EXPECT_EQ(out[0], in[1]);
            EXPECT_EQ(out[1], TYPE_T(0));
        }

        // 4. Overflow tolerance (n > Lanes)
        {
            ns::Batch<TYPE_T> v = ns::loadu_partial(in, Lanes + 10);
            ns::store(out, v);
            EXPECT_EQ(out[Lanes - 1], in[Lanes - 1]);
        }

        // load 0
        {
            FILL_ARRAY(in, TYPE_T(99));
            ns::Batch<TYPE_T> v = ns::loadu_partial(in, 0);
            FILL_ARRAY(out, TYPE_T(10));
            ns::storeu(out, v);
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(out[i] == 0);
            }
        }
        // store 0
        {
            FILL_ARRAY(in, TYPE_T(99));
            ns::Batch<TYPE_T> v = ns::load(in);
            FILL_ARRAY(out, TYPE_T(10));
            ns::storeu_partial(out, v, 0);
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(out[i] == 10);
            }
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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // 测试数据：验证位选择逻辑 (mask & a) | (~mask & b)
        TYPE_T val_a    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b10101));
        TYPE_T val_b    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b11111));
        TYPE_T val_mask = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b00010));
        uint_t expected = static_cast<uint_t>(0b11101);

        ns::store(res, ns::bit_if_then_else(ns::set(val_mask), ns::set(val_a), ns::set(val_b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(bit_equal(res[i], make_var_from_bits<TYPE_T>(expected)))
                << "Bit select failed at lane " << i
                << "\n  Expected bits: 0x" << std::hex << (uint64_t)expected
                << "\n  Actual bits:   0x" << (uint64_t)std::bit_cast<uint_t>(res[i]);
        }

        // 浮点数特殊验证：符号位搬运
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            TYPE_T pos_val = TYPE_T(1.0);
            TYPE_T neg_val = TYPE_T(-2.0);
            TYPE_T s_mask  = ksimd::SignBitMask<TYPE_T>;

            // 从 neg_val 取符号位，从 pos_val 取数值位，结果应为 -1.0
            ns::store(res, ns::bit_if_then_else(ns::set(s_mask), ns::set(neg_val), ns::set(pos_val)));

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
        
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        ns::Batch<TYPE_T> v_a = ns::set(TYPE_T(10));
        ns::Batch<TYPE_T> v_b = ns::set(TYPE_T(20));

        // 1. 全 1 掩码选择
        {
            auto mask_true = ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(1)));
            ns::store(res, ns::if_then_else(mask_true, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(10));
        }

        // 2. 全 0 掩码选择
        {
            auto mask_false = ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(2)));
            ns::store(res, ns::if_then_else(mask_false, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(20));
        }

        // 3. 混合掩码交叉选择
        {
            alignas(ns::Alignment<TYPE_T>) TYPE_T data_lhs[Lanes];
            alignas(ns::Alignment<TYPE_T>) TYPE_T data_rhs[Lanes];
            for (size_t i = 0; i < Lanes; ++i) {
                data_lhs[i] = static_cast<TYPE_T>(i);
                data_rhs[i] = TYPE_T(1);
            }

            auto mask_mixed = ns::greater(ns::load(data_lhs), ns::load(data_rhs));
            ns::store(res, ns::if_then_else(mask_mixed, v_a, v_b));

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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;

        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // 输入数据: ...010101 (0x15) -> 取反期望: ...101010 (低5位)
        uint_t input_bits = 0b10101;
        TYPE_T input_val = make_var_from_bits<TYPE_T>(static_cast<uint_t>(input_bits));

        ns::store(res, ns::bit_not(ns::set(input_val)));

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
            ns::store(res, ns::bit_not(ns::set(make_var_from_bits<TYPE_T>(zero_bits))));

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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;

        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10001
        uint_t a = 0b10101, b = 0b10011, exp = 0b10001;

        ns::store(res, ns::bit_and(ns::set(make_var_from_bits<TYPE_T>(a)),
                                   ns::set(make_var_from_bits<TYPE_T>(b))));

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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;

        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // 逻辑通常为: (~a) & b
        // a: 10101 (~a 低位: 01010)
        // b: 10011
        // res: 00010
        uint_t a = 0b10101, b = 0b10011, exp = 0b00010;

        ns::store(res, ns::bit_and_not(ns::set(make_var_from_bits<TYPE_T>(a)),
                                       ns::set(make_var_from_bits<TYPE_T>(b))));

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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;

        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10111
        uint_t a = 0b10101, b = 0b10011, exp = 0b10111;

        ns::store(res, ns::bit_or(ns::set(make_var_from_bits<TYPE_T>(a)),
                                  ns::set(make_var_from_bits<TYPE_T>(b))));

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
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;

        alignas(ns::Alignment<TYPE_T>) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 00110
        uint_t a = 0b10101, b = 0b10011, exp = 0b00110;

        ns::store(res, ns::bit_xor(ns::set(make_var_from_bits<TYPE_T>(a)),
                                   ns::set(make_var_from_bits<TYPE_T>(b))));

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
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(test, ns::add(ns::set(TYPE_T(10)), ns::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        // 浮点特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf + 1 = Inf
            ns::store(test, ns::add(ns::set(inf<TYPE_T>), ns::set(TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

            // NaN + 1 = NaN
            ns::store(test, ns::add(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf + (-Inf) = NaN
            ns::store(test, ns::add(ns::set(inf<TYPE_T>), ns::set(-inf<TYPE_T>)));
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
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(test, ns::sub(ns::set(TYPE_T(50)), ns::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf - Inf = NaN
            ns::store(test, ns::sub(ns::set(inf<TYPE_T>), ns::set(inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // 1.0 - NaN = NaN
            ns::store(test, ns::sub(ns::set(TYPE_T(1)), ns::set(qNaN<TYPE_T>)));
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
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(test, ns::mul(ns::set(TYPE_T(6)), ns::set(TYPE_T(7))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(42));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf * 0 = NaN
            ns::store(test, ns::mul(ns::set(inf<TYPE_T>), ns::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf * (-2) = -Inf
            ns::store(test, ns::mul(ns::set(inf<TYPE_T>), ns::set(TYPE_T(-2))));
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
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(test, ns::div(ns::set(TYPE_T(100)), ns::set(TYPE_T(4))));
        for (size_t i = 0; i < Lanes; ++i) {
            // 使用标准 EXPECT_NEAR 验证除法精度
            EXPECT_NEAR(static_cast<double>(test[i]), 25.0, 1e-7);
        }

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // 1.0 / 0.0 = Inf
            ns::store(test, ns::div(ns::set(TYPE_T(1)), ns::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

            // 0.0 / 0.0 = NaN
            ns::store(test, ns::div(ns::set(TYPE_T(0)), ns::set(TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf / Inf = NaN
            ns::store(test, ns::div(ns::set(inf<TYPE_T>), ns::set(inf<TYPE_T>)));
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
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T data[Lanes];
        TYPE_T expected = 0;
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
            expected += data[i];
        }

        TYPE_T res = ns::reduce_add(ns::load(data));
        EXPECT_NEAR((res), (expected), std::numeric_limits<TYPE_T>::epsilon() * 10);

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Inf in sum
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_add(ns::load(data))));

            // NaN in sum
            data[0] = qNaN<TYPE_T>;
            EXPECT_TRUE(std::isnan(ns::reduce_add(ns::load(data))));
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_add)
#endif

// ------------------------------------------ reduce_mul ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_mul() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T data[Lanes];

        // --- 1. 基础阶乘/累乘测试 ---
        TYPE_T expected = 1;
        for (size_t i = 0; i < Lanes; ++i) {
            // 使用较小的正数避免在 int8 或 float16 下过快溢出
            // 例如：1, 1, 2, 1, 1... 或者简单的交替
            data[i] = (i % 2 == 0) ? TYPE_T(2) : TYPE_T(1);
            expected *= data[i];
        }

        TYPE_T res = ns::reduce_mul(ns::load(data));
        
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            EXPECT_NEAR(res, expected, std::numeric_limits<TYPE_T>::epsilon() * 100);
        } else {
            EXPECT_EQ(res, expected);
        }

        // --- 2. 包含 0 的测试 (归零律) ---
        data[Lanes / 2] = TYPE_T(0);
        EXPECT_EQ(ns::reduce_mul(ns::load(data)), TYPE_T(0)) << "Multiplication by zero failed";

        // --- 3. 负数符号位测试 ---
        // 设置所有 lane 为 1，仅设置两个为 -1，结果应为 1
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            if constexpr (Lanes > 1)
            {
                for (size_t i = 0; i < Lanes; ++i) data[i] = TYPE_T(1);
                data[0] = TYPE_T(-1);
                data[1] = TYPE_T(-1);
                EXPECT_EQ(ns::reduce_mul(ns::load(data)), TYPE_T(1)) << "Double negative sign failed";
            }
        }

        // --- 4. 浮点数特殊值测试 ---
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Infinity 传播: inf * 2 = inf
            for (size_t i = 0; i < Lanes; ++i) data[i] = TYPE_T(2);
            data[Lanes / 2] = std::numeric_limits<TYPE_T>::infinity();
            EXPECT_TRUE(std::isinf(ns::reduce_mul(ns::load(data))));

            // NaN 传播: NaN * 1 = NaN
            data[Lanes / 2] = std::numeric_limits<TYPE_T>::quiet_NaN();
            EXPECT_TRUE(std::isnan(ns::reduce_mul(ns::load(data))));
            
            if constexpr (Lanes > 1)
            {
                // 0 * inf = NaN
                data[0] = TYPE_T(0);
                data[1] = std::numeric_limits<TYPE_T>::infinity();
                // 注意：某些架构优化可能导致结果不同，但 IEEE754 标准下应为 NaN
                EXPECT_TRUE(std::isnan(ns::reduce_mul(ns::load(data))));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_mul)
#endif

// ------------------------------------------ reduce_min ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_min() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T data[Lanes];

        // 1. 常规场景测试：[1, 2, 3, ..., Lanes]
        TYPE_T expected = TYPE_T(1);
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
        }
        TYPE_T res = ns::reduce_min<ksimd::FloatMinMaxOption::Native>(ns::load(data));
        EXPECT_EQ(res, expected);

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
        }
        res = ns::reduce_min<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data));
        EXPECT_EQ(res, expected);


        // 2. 最小值在末尾：[Lanes, Lanes-1, ..., 1]
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(Lanes - i);
        }
        res = ns::reduce_min<ksimd::FloatMinMaxOption::Native>(ns::load(data));
        EXPECT_EQ(res, TYPE_T(1));

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(Lanes - i);
        }
        res = ns::reduce_min<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data));
        EXPECT_EQ(res, TYPE_T(1));


        // 3. 包含负数
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes / 2] = TYPE_T(-100);
            res = ns::reduce_min<ksimd::FloatMinMaxOption::Native>(ns::load(data));
            EXPECT_EQ(res, TYPE_T(-100));

            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes / 2] = TYPE_T(-100);
            res = ns::reduce_min<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data));
            EXPECT_EQ(res, TYPE_T(-100));
        }

        // 4. 浮点数特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // 测试包含 -Inf (应为最小值)
            FILL_ARRAY(data, TYPE_T(0));
            data[0] = -inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_min(ns::load(data))) && ns::reduce_min(ns::load(data)) < 0);

            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes - 1] = -inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_min(ns::load(data))) && ns::reduce_min(ns::load(data)) < 0);

            // 测试 NaN 传播
            for (size_t i = 0; i < Lanes; ++i)
            {
                FILL_ARRAY(data, TYPE_T(0));
                data[i] = qNaN<TYPE_T>;
                EXPECT_TRUE(std::isnan(ns::reduce_min<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data))));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_min)
#endif

// ------------------------------------------ reduce_max ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_max() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T data[Lanes];

        TYPE_T res = TYPE_T(0);

        // 1. 全负数测试：[-Lanes, ..., -1]
        // 确保能正确识别较大的负数（如 -1 是最大值）
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            for (size_t i = 0; i < Lanes; ++i) {
                data[i] = -TYPE_T(Lanes - i);
            }
            res = ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data));
            EXPECT_EQ(res, TYPE_T(-1));

            for (size_t i = 0; i < Lanes; ++i) {
                data[i] = -TYPE_T(Lanes - i);
            }
            res = ns::reduce_max<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data));
            EXPECT_EQ(res, TYPE_T(-1));
        }

        // 2. 最大值在中间位置
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i);
        }
        data[Lanes / 2] = TYPE_T(999);
        res = ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data));
        EXPECT_EQ(res, TYPE_T(999));

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i);
        }
        data[Lanes / 2] = TYPE_T(999);
        res = ns::reduce_max<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data));
        EXPECT_EQ(res, TYPE_T(999));


        // 3. 浮点数特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // 测试正无穷 +Inf
            FILL_ARRAY(data, TYPE_T(0));
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data)))
                && ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data)) > 0);

            FILL_ARRAY(data, TYPE_T(0));
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data)))
                && ns::reduce_max<ksimd::FloatMinMaxOption::Native>(ns::load(data)) > 0);


            // 测试 NaN 传播
            for (size_t i = 0; i < Lanes; ++i)
            {
                FILL_ARRAY(data, TYPE_T(0));
                data[i] = qNaN<TYPE_T>;
                EXPECT_TRUE(std::isnan(ns::reduce_max<ksimd::FloatMinMaxOption::CheckNaN>(ns::load(data))));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_max)
#endif

// ------------------------------------------ mul_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul_add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        // 

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // (2 * 3) + 4 = 10
        ns::store(test, ns::mul_add(ns::set(TYPE_T(2)), ns::set(TYPE_T(3)), ns::set(TYPE_T(4))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN propagation
            ns::store(test, ns::mul_add(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(2)), ns::set(TYPE_T(3))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf propagation
            ns::store(test, ns::mul_add(ns::set(inf<TYPE_T>), ns::set(TYPE_T(2)), ns::set(TYPE_T(3))));
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        ns::store(test, ns::min(ns::set(TYPE_T(10)), ns::set(TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Min(Inf, 100) = 100
            ns::store(test, ns::min(ns::set(inf<TYPE_T>), ns::set(TYPE_T(100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // Min(100, Inf) = 100
            ns::store(test, ns::min(ns::set(TYPE_T(100)), ns::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // NaN 行为 (依照指令集惯例，通常返回非 NaN 操作数，或者取决于位置)
            ns::store(test, ns::min(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(5))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(5)));

            // 右操作数是 NaN: 返回 NaN
            ns::store(test, ns::min(ns::set(TYPE_T(5)), ns::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }

            // Check 模式，无论左右，都返回NaN
            ns::store(test, ns::min<ksimd::FloatMinMaxOption::CheckNaN>(ns::set(TYPE_T(5)), ns::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
            ns::store(test, ns::min<ksimd::FloatMinMaxOption::CheckNaN>(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(5))));
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        ns::store(test, ns::max(ns::set(TYPE_T(10)), ns::set(TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(20)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Max(-Inf, -100) = -100
            ns::store(test, ns::max(ns::set(-inf<TYPE_T>), ns::set(TYPE_T(-100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));

            // NaN 行为
            ns::store(test, ns::max(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(-100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));

            ns::store(test, ns::max(ns::set(TYPE_T(-100)), ns::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }

            // Check 模式，无论左右，都返回NaN
            ns::store(test, ns::max<ksimd::FloatMinMaxOption::CheckNaN>(ns::set(TYPE_T(5)), ns::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
            ns::store(test, ns::max<ksimd::FloatMinMaxOption::CheckNaN>(ns::set(qNaN<TYPE_T>), ns::set(TYPE_T(5))));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 1 == 1 (True)
        ns::test_store_mask(test, ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        // 1 == 2 (False)
        ns::test_store_mask(test, ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN == NaN (False)
            ns::test_store_mask(test, ns::equal(ns::set(qNaN<TYPE_T>), ns::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void not_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 1 != 2 (True)
        ns::test_store_mask(test, ns::not_equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN != NaN (True)
            ns::test_store_mask(test, ns::not_equal(ns::set(qNaN<TYPE_T>), ns::set(qNaN<TYPE_T>)));
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 2 > 1 (True), 1 > 2 (False)
        ns::test_store_mask(test, ns::greater(ns::set(TYPE_T(2)), ns::set(TYPE_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Inf > 1e30 (True)
            ns::test_store_mask(test, ns::greater(ns::set(inf<TYPE_T>), ns::set(TYPE_T(1e30))));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // 1 > -Inf (True)
            ns::test_store_mask(test, ns::greater(ns::set(TYPE_T(1)), ns::set(-inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN > 任何数 (False)
            ns::test_store_mask(test, ns::greater(ns::set(qNaN<TYPE_T>), ns::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void greater_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 2 >= 2 (True)
        ns::test_store_mask(test, ns::greater_equal(ns::set(TYPE_T(2)), ns::set(TYPE_T(2))));
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
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 1 < 2 (True)
        ns::test_store_mask(test, ns::less(ns::set(TYPE_T(1)), ns::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // -Inf < Inf (True)
            ns::test_store_mask(test, ns::less(ns::set(-inf<TYPE_T>), ns::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN 相关比较应为 False
            ns::test_store_mask(test, ns::less(ns::set(-inf<TYPE_T>), ns::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

            ns::test_store_mask(test, ns::less(ns::set(qNaN<TYPE_T>), ns::set(-inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void less_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T test[Lanes];

        // 5 <= 5 (True)
        ns::test_store_mask(test, ns::less_equal(ns::set(TYPE_T(5)), ns::set(TYPE_T(5))));
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
        
        
        using mask_t = ns::Mask<TYPE_T>;

        // 准备测试数据
        ns::Batch<TYPE_T> v1 = ns::set(static_cast<TYPE_T>(10));
        ns::Batch<TYPE_T> v2 = ns::set(static_cast<TYPE_T>(20));

        mask_t m_true  = ns::equal(v1, v1);  // All Ones
        mask_t m_false = ns::equal(v1, v2);  // All Zeros

        #define KSIMD_CHECK_MASK_EQ(lhs, rhs) \
        do { \
            alignas(ns::Alignment<TYPE_T>) TYPE_T M__l[ns::Lanes<TYPE_T>]{}; \
            alignas(ns::Alignment<TYPE_T>) TYPE_T M__r[ns::Lanes<TYPE_T>]{}; \
            ns::test_store_mask(M__l, lhs); \
            ns::test_store_mask(M__r, rhs); \
            for (size_t I__ = 0; I__ < ns::Lanes<TYPE_T>; ++I__) \
            { \
                EXPECT_TRUE(bit_equal(M__l[I__], M__r[I__])); \
            } \
        } while (0)

        // 1. 基础位运算函数测试 (and, or, xor, not)
        {
            KSIMD_CHECK_MASK_EQ(ns::mask_and(m_true, m_false), m_false);
            KSIMD_CHECK_MASK_EQ(ns::mask_or(m_true, m_false),  m_true);
            KSIMD_CHECK_MASK_EQ(ns::mask_xor(m_true, m_true),  m_false);
            KSIMD_CHECK_MASK_EQ(ns::mask_not(m_true),          m_false);
            KSIMD_CHECK_MASK_EQ(ns::mask_not(m_false),         m_true);
        }
        #undef KSIMD_CHECK_MASK_EQ
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mask_logic)
#endif

// ------------------------------------------ test_arithmetic_forwarding ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void test_arithmetic_forwarding() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T act[Lanes], exp[Lanes];

        #define check(actual, expected, msg) \
            do { \
                ns::store(act, actual); \
                ns::store(exp, expected); \
                for (size_t i = 0; i < Lanes; ++i)  \
                    EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Forwarding failed: " << msg << " at lane " << i; \
            } while (0)

        ns::Batch<TYPE_T> a = ns::set(TYPE_T(10.5)), b = ns::set(TYPE_T(2.0));

        // 二元运算
        check(a + b, ns::add(a, b), "operator+");
        check(a - b, ns::sub(a, b), "operator-");
        check(a * b, ns::mul(a, b), "operator*");
        check(a / b, ns::div(a, b), "operator/");

        // 一元与赋值
        check(-a, ns::neg(a), "unary operator-");

        ns::Batch<TYPE_T> c = a; c += b; check(c, ns::add(a, b), "operator+=");
        c = a; c -= b;         check(c, ns::sub(a, b), "operator-=");
        c = a; c *= b;         check(c, ns::mul(a, b), "operator*=");
        c = a; c /= b;         check(c, ns::div(a, b), "operator/=");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(test_arithmetic_forwarding)
#endif

// ------------------------------------------ test_bitwise_forwarding ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void test_bitwise_forwarding() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>; // 自动获取对应宽度的无符号整型

        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T act[Lanes], exp[Lanes];

        #define check(actual, expected, msg) \
        do { \
            ns::store(act, actual); \
            ns::store(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        // 使用全量程的交替位模式
        // 如果是 32位，它会截断为 0xAAAA5555
        // 如果是 64位，它是 0xAAAAAAAAAAAAAAAAULL (取决于你的 uint_t 定义)
        uint_t p1 = static_cast<uint_t>(0xAAAAAAAAAAAAAAAAULL);
        uint_t p2 = static_cast<uint_t>(0x5555555555555555ULL);

        ns::Batch<TYPE_T> a = ns::set(std::bit_cast<TYPE_T>(p1));
        ns::Batch<TYPE_T> b = ns::set(std::bit_cast<TYPE_T>(p2));

        // 1. 基础位运算重载转发
        check(a & b, ns::bit_and(a, b), "operator&");
        check(a | b, ns::bit_or(a, b),  "operator|");
        check(a ^ b, ns::bit_xor(a, b), "operator^");
        check(~a,    ns::bit_not(a),   "operator~");

        // 2. 复合赋值重载转发
        ns::Batch<TYPE_T> c = a;
        c &= b; check(c, ns::bit_and(a, b), "operator&=");

        c = a;
        c |= b; check(c, ns::bit_or(a, b),  "operator|=");

        c = a;
        c ^= b; check(c, ns::bit_xor(a, b), "operator^=");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(test_bitwise_forwarding)
#endif

// ------------------------------------------ test_comparison_forwarding ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void test_comparison_forwarding() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        
        // using mask_t = ns::Mask<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T act[Lanes], exp[Lanes];

        #define check(actual, expected, msg) \
        do { \
            ns::test_store_mask(act, actual); \
            ns::test_store_mask(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        ns::Batch<TYPE_T> a = ns::set(TYPE_T(10)), b = ns::set(TYPE_T(20));

        check(a == b, ns::equal(a, b),         "operator==");
        check(a != b, ns::not_equal(a, b),     "operator!=");
        check(a <  b, ns::less(a, b),          "operator<");
        check(a <= b, ns::less_equal(a, b),    "operator<=");
        check(a >  b, ns::greater(a, b),       "operator>");
        check(a >= b, ns::greater_equal(a, b), "operator>=");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(test_comparison_forwarding)
#endif

// ------------------------------------------ test_mixed_forwarding ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void test_mixed_forwarding() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        
        // using mask_t = ns::Mask<TYPE_T>;
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T act[Lanes], exp[Lanes];

        const ns::Batch<TYPE_T> b = ns::set(TYPE_T(10));
        const TYPE_T s = TYPE_T(2);
        const ns::Batch<TYPE_T> bs = ns::set(s);

        // --- 辅助宏：验证 Batch 结果 ---
        #define KSIMD_CHECK_B_MIXED(actual, expected, msg) \
        do { \
            ns::store(act, actual); ns::store(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Mixed Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        // --- 辅助宏：验证 Mask 结果 ---
        #define KSIMD_CHECK_M_MIXED(actual, expected, msg) \
        do { \
            ns::test_store_mask(act, actual); ns::test_store_mask(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Mixed Mask Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        // 1. 混合算术运算转发 (+, -, *, /)
        KSIMD_CHECK_B_MIXED(b + s, ns::add(b, bs), "Batch + Scalar");
        KSIMD_CHECK_B_MIXED(s + b, ns::add(bs, b), "Scalar + Batch");

        KSIMD_CHECK_B_MIXED(b - s, ns::sub(b, bs), "Batch - Scalar");
        KSIMD_CHECK_B_MIXED(s - b, ns::sub(bs, b), "Scalar - Batch");

        KSIMD_CHECK_B_MIXED(b * s, ns::mul(b, bs), "Batch * Scalar");
        KSIMD_CHECK_B_MIXED(s * b, ns::mul(bs, b), "Scalar * Batch");

        KSIMD_CHECK_B_MIXED(b / s, ns::div(b, bs), "Batch / Scalar");
        KSIMD_CHECK_B_MIXED(s / b, ns::div(bs, b), "Scalar / Batch");

        // 2. 混合位运算转发 (&, |, ^)
        // 使用 bit_cast 构造一个特定的位模式标量进行测试
        const TYPE_T s_bit = std::bit_cast<TYPE_T>(static_cast<uint_t>(0x0F0F0F0F0F0F0F0FULL));
        const ns::Batch<TYPE_T> bs_bit = ns::set(s_bit);

        KSIMD_CHECK_B_MIXED(b & s_bit, ns::bit_and(b, bs_bit), "Batch & Scalar");
        KSIMD_CHECK_B_MIXED(s_bit & b, ns::bit_and(bs_bit, b), "Scalar & Batch");

        KSIMD_CHECK_B_MIXED(b | s_bit, ns::bit_or(b, bs_bit),  "Batch | Scalar");
        KSIMD_CHECK_B_MIXED(s_bit | b, ns::bit_or(bs_bit, b),  "Scalar | Batch");

        KSIMD_CHECK_B_MIXED(b ^ s_bit, ns::bit_xor(b, bs_bit), "Batch ^ Scalar");
        KSIMD_CHECK_B_MIXED(s_bit ^ b, ns::bit_xor(bs_bit, b), "Scalar ^ Batch");

        // 2. 混合复合赋值转发 (+=, -=, *=, /=)
        {
            ns::Batch<TYPE_T> c = b; c += s; KSIMD_CHECK_B_MIXED(c, ns::add(b, bs), "Batch += Scalar");
            c = b;         c -= s; KSIMD_CHECK_B_MIXED(c, ns::sub(b, bs), "Batch -= Scalar");
            c = b;         c *= s; KSIMD_CHECK_B_MIXED(c, ns::mul(b, bs), "Batch *= Scalar");
            c = b;         c /= s; KSIMD_CHECK_B_MIXED(c, ns::div(b, bs), "Batch /= Scalar");

            c = b; c &= s_bit; KSIMD_CHECK_B_MIXED(c, ns::bit_and(b, bs_bit), "Batch &= Scalar");
            c = b; c |= s_bit; KSIMD_CHECK_B_MIXED(c, ns::bit_or(b, bs_bit),  "Batch |= Scalar");
            c = b; c ^= s_bit; KSIMD_CHECK_B_MIXED(c, ns::bit_xor(b, bs_bit), "Batch ^= Scalar");
        }

        // 3. 混合比较运算转发 (==, !=, <, <=, >, >=)
        KSIMD_CHECK_M_MIXED(b == s, ns::equal(b, bs),         "Batch == Scalar");
        KSIMD_CHECK_M_MIXED(s == b, ns::equal(bs, b),         "Scalar == Batch");

        KSIMD_CHECK_M_MIXED(b != s, ns::not_equal(b, bs),     "Batch != Scalar");
        KSIMD_CHECK_M_MIXED(s != b, ns::not_equal(bs, b),     "Scalar != Batch");

        KSIMD_CHECK_M_MIXED(b < s,  ns::less(b, bs),          "Batch < Scalar");
        KSIMD_CHECK_M_MIXED(s < b,  ns::less(bs, b),          "Scalar < Batch");

        KSIMD_CHECK_M_MIXED(b <= s, ns::less_equal(b, bs),    "Batch <= Scalar");
        KSIMD_CHECK_M_MIXED(s <= b, ns::less_equal(bs, b),    "Scalar <= Batch");

        KSIMD_CHECK_M_MIXED(b > s,  ns::greater(b, bs),       "Batch > Scalar");
        KSIMD_CHECK_M_MIXED(s > b,  ns::greater(bs, b),       "Scalar > Batch");

        KSIMD_CHECK_M_MIXED(b >= s, ns::greater_equal(b, bs), "Batch >= Scalar");
        KSIMD_CHECK_M_MIXED(s >= b, ns::greater_equal(bs, b), "Scalar >= Batch");

        #undef KSIMD_CHECK_B_MIXED
        #undef KSIMD_CHECK_M_MIXED
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(test_mixed_forwarding)
#endif

// ------------------------------------------ test_mask_forwarding ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void test_mask_forwarding() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        using mask_t = ns::Mask<TYPE_T>;
        constexpr size_t Lanes = ns::Lanes<TYPE_T>;
        alignas(ns::Alignment<TYPE_T>) TYPE_T act[Lanes], exp[Lanes];

        #define check(actual, expected, msg) \
        do { \
            ns::test_store_mask(act, actual); \
            ns::test_store_mask(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(bit_equal(act[i], exp[i])) << "Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        mask_t m1 = ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(1))); // True
        mask_t m2 = ns::equal(ns::set(TYPE_T(1)), ns::set(TYPE_T(2))); // False

        check(m1 & m2, ns::mask_and(m1, m2), "mask operator&");
        check(m1 | m2, ns::mask_or(m1, m2),  "mask operator|");
        check(m1 ^ m2, ns::mask_xor(m1, m2), "mask operator^");
        check(~m1,     ns::mask_not(m1),     "mask operator~");

        mask_t mc = m1; mc &= m2; check(mc, ns::mask_and(m1, m2), "mask operator&=");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(test_mask_forwarding)
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

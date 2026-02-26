// using TYPE_T = uint32_t;

#if defined(_WIN32) || defined(WIN64)
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

#include <kSimd/core/aligned_allocate.hpp>
#include <vector>

KSIMD_WARNING_PUSH
KSIMD_IGNORE_WARNING_MSVC(4723) // ignore warning: divide by 0

// ------------------------------------------ undefined ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void undefined() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        TAG_T t;

        [[maybe_unused]] ns::Batch<decltype(t)> z = ns::undefined(t);
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> arr(Lanes, TYPE_T(0xff));

        ns::Batch<decltype(t)> z = ns::zero(t);
        ns::store(t, arr.data(), z);
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> arr(Lanes);

        // 测试常规数值广播
        TYPE_T val = TYPE_T(42);
        ns::Batch<decltype(t)> v = ns::set(t, val);
        ns::store(t, arr.data(), v);
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(arr[i], val);

        // 针对浮点数的特殊值测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN 广播
            ns::store(t, arr.data(), ns::set(t, qNaN<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(arr[i]));

            // Inf 广播
            ns::store(t, arr.data(), ns::set(t, inf<TYPE_T>));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> arr(Lanes);

        // 1. 无参 sequence(): [0, 1, 2, ...]
        ns::store(t, arr.data(), ns::sequence(t));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(i));
        }

        // 2. 带 base: [base, base + 1, ...]
        TYPE_T base = TYPE_T(10);
        ns::store(t, arr.data(), ns::sequence(t, base));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(base + static_cast<TYPE_T>(i)));
        }

        // 3. 带 base 和 stride: [base, base + stride, ...]
        TYPE_T b_v = TYPE_T(5), stride = TYPE_T(2);
        ns::store(t, arr.data(), ns::sequence(t, b_v, stride));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(arr[i], static_cast<TYPE_T>(b_v + static_cast<TYPE_T>(i) * stride));
        }

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            TYPE_T f_base = TYPE_T(1.5), f_stride = TYPE_T(-0.5);
            ns::store(t, arr.data(), ns::sequence(t, f_base, f_stride));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> in(Lanes);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> out(Lanes);

        for (size_t i = 0; i < Lanes; ++i) {
            in[i] = TYPE_T(i + 7);
            out[i] = TYPE_T(0);
        }

        ns::store(t, out.data(), ns::load(t, in.data()));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        const size_t Lanes = ns::lanes(t);
        // 分配略大空间以模拟非对齐
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> buffer_in(Lanes + 1);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> buffer_out(Lanes + 1);

        TYPE_T* u_in = buffer_in.data() + 1;
        TYPE_T* u_out = buffer_out.data() + 1;

        for (size_t i = 0; i < Lanes; ++i) {
            u_in[i] = TYPE_T(i * 3 + 1);
            u_out[i] = TYPE_T(0);
        }

        ns::storeu(t, u_out, ns::loadu(t, u_in));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> in(Lanes * 2);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> out(Lanes * 2);

        for (size_t i = 0; i < Lanes * 2; ++i) in[i] = TYPE_T(i + 1);

        // 1. loadu_partial & zero-padding check
        for (size_t n = 0; n <= Lanes; ++n) {
            std::memset(out.data(), 0xAA, sizeof(TYPE_T) * out.size()); // 干扰值
            ns::Batch<decltype(t)> v = ns::loadu_partial(t, in.data(), n);
            ns::store(t, out.data(), v);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], in[i]);
                else EXPECT_EQ(out[i], TYPE_T(0)); // 必须清零
            }
        }

        // 2. storeu_partial & memory protection
        for (size_t n = 0; n <= Lanes; ++n) {
            constexpr TYPE_T sentinel = TYPE_T(88);
            for (size_t i = 0; i < Lanes; ++i) out[i] = sentinel;

            ns::Batch<decltype(t)> v = ns::set(t, TYPE_T(99));
            ns::storeu_partial(t, out.data(), v, n);

            for (size_t i = 0; i < Lanes; ++i) {
                if (i < n) EXPECT_EQ(out[i], TYPE_T(99));
                else EXPECT_EQ(out[i], sentinel); // 不应触碰
            }
        }

        // 3. Unaligned safety
        if constexpr (Lanes > 1) {
            ns::Batch<decltype(t)> v = ns::loadu_partial(t, in.data() + 1, 1);
            ns::store(t, out.data(), v);
            EXPECT_EQ(out[0], in[1]);
            EXPECT_EQ(out[1], TYPE_T(0));
        }

        // 4. Overflow tolerance (n > Lanes)
        {
            ns::Batch<decltype(t)> v = ns::loadu_partial(t, in.data(), Lanes + 10);
            ns::store(t, out.data(), v);
            EXPECT_EQ(out[Lanes - 1], in[Lanes - 1]);
        }

        // load 0
        {
            FILL_ARRAY(in, TYPE_T(99));
            ns::Batch<decltype(t)> v = ns::loadu_partial(t, in.data(), 0);
            FILL_ARRAY(out, TYPE_T(10));
            ns::storeu(t, out.data(), v);
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(out[i] == 0);
            }
        }
        // store 0
        {
            FILL_ARRAY(in, TYPE_T(99));
            ns::Batch<decltype(t)> v = ns::load(t, in.data());
            FILL_ARRAY(out, TYPE_T(10));
            ns::storeu_partial(t, out.data(), v, 0);
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // 测试数据：验证位选择逻辑 (mask & a) | (~mask & b)
        TYPE_T val_a    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b10101));
        TYPE_T val_b    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b11111));
        TYPE_T val_mask = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b00010));
        uint_t expected = static_cast<uint_t>(0b11101);

        ns::store(t, res, ns::bit_if_then_else(t, ns::set(t, val_mask), ns::set(t, val_a), ns::set(t, val_b)));

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
            ns::store(t, res, ns::bit_if_then_else(t, ns::set(t, s_mask), ns::set(t, neg_val), ns::set(t, pos_val)));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T res[Lanes];

        ns::Batch<decltype(t)> v_a = ns::set(t, TYPE_T(10));
        ns::Batch<decltype(t)> v_b = ns::set(t, TYPE_T(20));

        // 1. 全 1 掩码选择
        {
            auto mask_true = ns::equal(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(1)));
            ns::store(t, res, ns::if_then_else(t, mask_true, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(10));
        }

        // 2. 全 0 掩码选择
        {
            auto mask_false = ns::equal(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(2)));
            ns::store(t, res, ns::if_then_else(t, mask_false, v_a, v_b));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(res[i], TYPE_T(20));
        }

        // 3. 混合掩码交叉选择
        {
            alignas(ALIGNMENT) TYPE_T data_lhs[Lanes];
            alignas(ALIGNMENT) TYPE_T data_rhs[Lanes];
            for (size_t i = 0; i < Lanes; ++i) {
                data_lhs[i] = static_cast<TYPE_T>(i);
                data_rhs[i] = TYPE_T(1);
            }

            auto mask_mixed = ns::greater(t, ns::load(t, data_lhs), ns::load(t, data_rhs));
            ns::store(t, res, ns::if_then_else(t, mask_mixed, v_a, v_b));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        const size_t Lanes = ns::lanes(t);

        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // 输入数据: ...010101 (0x15) -> 取反期望: ...101010 (低5位)
        uint_t input_bits = 0b10101;
        TYPE_T input_val = make_var_from_bits<TYPE_T>(static_cast<uint_t>(input_bits));

        ns::store(t, res, ns::bit_not(t, ns::set(t, input_val)));

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
            ns::store(t, res, ns::bit_not(t, ns::set(t, make_var_from_bits<TYPE_T>(zero_bits))));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        const size_t Lanes = ns::lanes(t);

        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10001
        uint_t a = 0b10101, b = 0b10011, exp = 0b10001;

        ns::store(t, res, ns::bit_and(t, ns::set(t, make_var_from_bits<TYPE_T>(a)),
                                   ns::set(t, make_var_from_bits<TYPE_T>(b))));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        const size_t Lanes = ns::lanes(t);

        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // 逻辑通常为: (~a) & b
        // a: 10101 (~a 低位: 01010)
        // b: 10011
        // res: 00010
        uint_t a = 0b10101, b = 0b10011, exp = 0b00010;

        ns::store(t, res, ns::bit_and_not(t, ns::set(t, make_var_from_bits<TYPE_T>(a)),
                                       ns::set(t, make_var_from_bits<TYPE_T>(b))));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        const size_t Lanes = ns::lanes(t);

        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 10111
        uint_t a = 0b10101, b = 0b10011, exp = 0b10111;

        ns::store(t, res, ns::bit_or(t, ns::set(t, make_var_from_bits<TYPE_T>(a)),
                                  ns::set(t, make_var_from_bits<TYPE_T>(b))));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        using uint_t = ksimd::same_bits_uint_t<TYPE_T>;
        const size_t Lanes = ns::lanes(t);

        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // a: 10101, b: 10011 -> res: 00110
        uint_t a = 0b10101, b = 0b10011, exp = 0b00110;

        ns::store(t, res, ns::bit_xor(t, ns::set(t, make_var_from_bits<TYPE_T>(a)),
                                   ns::set(t, make_var_from_bits<TYPE_T>(b))));

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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(t, test, ns::add(t, ns::set(t, TYPE_T(10)), ns::set(t, TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        // 浮点特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf + 1 = Inf
            ns::store(t, test, ns::add(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);

            // NaN + 1 = NaN
            ns::store(t, test, ns::add(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(1))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf + (-Inf) = NaN
            ns::store(t, test, ns::add(t, ns::set(t, inf<TYPE_T>), ns::set(t, -inf<TYPE_T>)));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(t, test, ns::sub(t, ns::set(t, TYPE_T(50)), ns::set(t, TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(30));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf - Inf = NaN
            ns::store(t, test, ns::sub(t, ns::set(t, inf<TYPE_T>), ns::set(t, inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // 1.0 - NaN = NaN
            ns::store(t, test, ns::sub(t, ns::set(t, TYPE_T(1)), ns::set(t, qNaN<TYPE_T>)));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        // 常规数值测试
        ns::store(t, test, ns::mul(t, ns::set(t, TYPE_T(6)), ns::set(t, TYPE_T(7))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(42));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>)
        {
            // Inf * 0 = NaN
            ns::store(t, test, ns::mul(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(0))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf * (-2) = -Inf
            ns::store(t, test, ns::mul(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(-2))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] < 0);
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(mul)
#endif

// ------------------------------------------ reduce_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T data[Lanes];
        TYPE_T expected = 0;
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
            expected += data[i];
        }

        TYPE_T res = ns::reduce_add(t, ns::load(t, data));
        EXPECT_NEAR((res), (expected), std::numeric_limits<TYPE_T>::epsilon() * 10);

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Inf in sum
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_add(t, ns::load(t, data))));

            // NaN in sum
            data[0] = qNaN<TYPE_T>;
            EXPECT_TRUE(std::isnan(ns::reduce_add(t, ns::load(t, data))));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T data[Lanes];

        // --- 1. 基础阶乘/累乘测试 ---
        TYPE_T expected = 1;
        for (size_t i = 0; i < Lanes; ++i) {
            // 使用较小的正数避免在 int8 或 float16 下过快溢出
            // 例如：1, 1, 2, 1, 1... 或者简单的交替
            data[i] = (i % 2 == 0) ? TYPE_T(2) : TYPE_T(1);
            expected *= data[i];
        }

        TYPE_T res = ns::reduce_mul(t, ns::load(t, data));
        
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            EXPECT_NEAR(res, expected, std::numeric_limits<TYPE_T>::epsilon() * 100);
        } else {
            EXPECT_EQ(res, expected);
        }

        // --- 2. 包含 0 的测试 (归零律) ---
        data[Lanes / 2] = TYPE_T(0);
        EXPECT_EQ(ns::reduce_mul(t, ns::load(t, data)), TYPE_T(0)) << "Multiplication by zero failed";

        // --- 3. 负数符号位测试 ---
        // 设置所有 lane 为 1，仅设置两个为 -1，结果应为 1
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            if constexpr (Lanes > 1)
            {
                for (size_t i = 0; i < Lanes; ++i) data[i] = TYPE_T(1);
                data[0] = TYPE_T(-1);
                data[1] = TYPE_T(-1);
                EXPECT_EQ(ns::reduce_mul(t, ns::load(t, data)), TYPE_T(1)) << "Double negative sign failed";
            }
        }

        // --- 4. 浮点数特殊值测试 ---
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Infinity 传播: inf * 2 = inf
            for (size_t i = 0; i < Lanes; ++i) data[i] = TYPE_T(2);
            data[Lanes / 2] = std::numeric_limits<TYPE_T>::infinity();
            EXPECT_TRUE(std::isinf(ns::reduce_mul(t, ns::load(t, data))));

            // NaN 传播: NaN * 1 = NaN
            data[Lanes / 2] = std::numeric_limits<TYPE_T>::quiet_NaN();
            EXPECT_TRUE(std::isnan(ns::reduce_mul(t, ns::load(t, data))));
            
            if constexpr (Lanes > 1)
            {
                // 0 * inf = NaN
                data[0] = TYPE_T(0);
                data[1] = std::numeric_limits<TYPE_T>::infinity();
                // 注意：某些架构优化可能导致结果不同，但 IEEE754 标准下应为 NaN
                EXPECT_TRUE(std::isnan(ns::reduce_mul(t, ns::load(t, data))));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T data[Lanes];

        // 1. 常规场景测试：[1, 2, 3, ..., Lanes]
        TYPE_T expected = TYPE_T(1);
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
        }
        TYPE_T res = ns::reduce_min<ns::FloatMinMaxOption::Native>(t, ns::load(t, data));
        EXPECT_EQ(res, expected);

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
        }
        res = ns::reduce_min<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data));
        EXPECT_EQ(res, expected);


        // 2. 最小值在末尾：[Lanes, Lanes-1, ..., 1]
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(Lanes - i);
        }
        res = ns::reduce_min<ns::FloatMinMaxOption::Native>(t, ns::load(t, data));
        EXPECT_EQ(res, TYPE_T(1));

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(Lanes - i);
        }
        res = ns::reduce_min<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data));
        EXPECT_EQ(res, TYPE_T(1));


        // 3. 包含负数
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes / 2] = TYPE_T(-100);
            res = ns::reduce_min<ns::FloatMinMaxOption::Native>(t, ns::load(t, data));
            EXPECT_EQ(res, TYPE_T(-100));

            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes / 2] = TYPE_T(-100);
            res = ns::reduce_min<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data));
            EXPECT_EQ(res, TYPE_T(-100));
        }

        // 4. 浮点数特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // 测试包含 -Inf (应为最小值)
            FILL_ARRAY(data, TYPE_T(0));
            data[0] = -inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_min(t, ns::load(t, data))) && ns::reduce_min(t, ns::load(t, data)) < 0);

            FILL_ARRAY(data, TYPE_T(0));
            data[Lanes - 1] = -inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_min(t, ns::load(t, data))) && ns::reduce_min(t, ns::load(t, data)) < 0);

            // 测试 NaN 传播
            for (size_t i = 0; i < Lanes; ++i)
            {
                FILL_ARRAY(data, TYPE_T(0));
                data[i] = qNaN<TYPE_T>;
                EXPECT_TRUE(std::isnan(ns::reduce_min<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data))));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T data[Lanes];

        TYPE_T res = TYPE_T(0);

        // 1. 全负数测试：[-Lanes, ..., -1]
        // 确保能正确识别较大的负数（如 -1 是最大值）
        if constexpr (ksimd::is_scalar_signed<TYPE_T>)
        {
            for (size_t i = 0; i < Lanes; ++i) {
                data[i] = -TYPE_T(Lanes - i);
            }
            res = ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data));
            EXPECT_EQ(res, TYPE_T(-1));

            for (size_t i = 0; i < Lanes; ++i) {
                data[i] = -TYPE_T(Lanes - i);
            }
            res = ns::reduce_max<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data));
            EXPECT_EQ(res, TYPE_T(-1));
        }

        // 2. 最大值在中间位置
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i);
        }
        data[Lanes / 2] = TYPE_T(999);
        res = ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data));
        EXPECT_EQ(res, TYPE_T(999));

        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i);
        }
        data[Lanes / 2] = TYPE_T(999);
        res = ns::reduce_max<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data));
        EXPECT_EQ(res, TYPE_T(999));


        // 3. 浮点数特殊边界测试
        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // 测试正无穷 +Inf
            FILL_ARRAY(data, TYPE_T(0));
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data)))
                && ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data)) > 0);

            FILL_ARRAY(data, TYPE_T(0));
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data)))
                && ns::reduce_max<ns::FloatMinMaxOption::Native>(t, ns::load(t, data)) > 0);


            // 测试 NaN 传播
            for (size_t i = 0; i < Lanes; ++i)
            {
                FILL_ARRAY(data, TYPE_T(0));
                data[i] = qNaN<TYPE_T>;
                EXPECT_TRUE(std::isnan(ns::reduce_max<ns::FloatMinMaxOption::CheckNaN>(t, ns::load(t, data))));
            }
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_max)
#endif

// ------------------------------------------ reduce_mask ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_mask() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        TAG_T t;

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T data[Lanes];

        using MaskBitset = ns::MaskBitset<TAG_T>;

        // 1. 全 0 测试：所有掩码均为 false
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(0);
        }
        // 假设这里通过比较生成掩码，例如：data > 0
        auto mask_none = ns::greater(t, ns::load(t, data), ns::zero(t));
        MaskBitset res_none = ns::reduce_mask(t, mask_none);
        EXPECT_TRUE(first_n_bit_false(res_none, Lanes));

        // 2. 全 1 测试：所有掩码均为 true
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(1);
        }
        auto mask_all = ns::greater(t, ns::load(t, data), ns::zero(t));
        MaskBitset res_all = ns::reduce_mask(t, mask_all);
        // 对于 32位 128bit (4 lanes)，全 1 应该是 0b1111 (即 15)
        EXPECT_TRUE(first_n_bit_true(res_all, Lanes));

        // 3. 逐通道测试：依次激活每一个通道
        for (size_t i = 0; i < Lanes; ++i) {
            // 清空数组
            for (size_t j = 0; j < Lanes; ++j) data[j] = TYPE_T(-1.0);
            // 仅令第 i 个元素大于 0
            data[i] = TYPE_T(1.0);

            auto mask_single = ns::greater(t, ns::load(t, data), ns::zero(t));
            MaskBitset res_single = ns::reduce_mask(t, mask_single);

            // 结果应该是第 i 位被置 1
            EXPECT_EQ(res_single, MaskBitset(1ULL << i));
        }

        // 4. 交替位测试：0101...
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = (i % 2 == 0) ? TYPE_T(1) : TYPE_T(-1);
        }
        auto mask_alt = ns::greater(t, ns::load(t, data), ns::zero(t));
        MaskBitset res_alt = ns::reduce_mask(t, mask_alt);

        MaskBitset expected_alt = 0;
        for (size_t i = 0; i < Lanes; i += 2) expected_alt |= (1ULL << i);

        EXPECT_EQ(res_alt, expected_alt);
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_mask)
#endif

// ------------------------------------------ mul_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul_add() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        // 

        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        // (2 * 3) + 4 = 10
        ns::store(t, test, ns::mul_add(t, ns::set(t, TYPE_T(2)), ns::set(t, TYPE_T(3)), ns::set(t, TYPE_T(4))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN propagation
            ns::store(t, test, ns::mul_add(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(2)), ns::set(t, TYPE_T(3))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf propagation
            ns::store(t, test, ns::mul_add(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(2)), ns::set(t, TYPE_T(3))));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        ns::store(t, test, ns::min(t, ns::set(t, TYPE_T(10)), ns::set(t, TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Min(Inf, 100) = 100
            ns::store(t, test, ns::min(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // Min(100, Inf) = 100
            ns::store(t, test, ns::min(t, ns::set(t, TYPE_T(100)), ns::set(t, inf<TYPE_T>)));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(100)));

            // NaN 行为 (不同指令行为不同，不进行测试)
            // ns::store(t, test, ns::min(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(5))));
            // EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(5)));

            // // 右操作数是 NaN: 返回 NaN
            // ns::store(t, test, ns::min(t, ns::set(t, TYPE_T(5)), ns::set(t, qNaN<TYPE_T>)));
            // for (size_t i = 0; i < Lanes; ++i)
            // {
            //     EXPECT_TRUE(std::isnan(test[i]));
            // }

            // Check 模式，无论左右，都返回NaN
            ns::store(t, test, ns::min<ns::FloatMinMaxOption::CheckNaN>(t, ns::set(t, TYPE_T(5)), ns::set(t, qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
            ns::store(t, test, ns::min<ns::FloatMinMaxOption::CheckNaN>(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(5))));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void max() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        alignas(ALIGNMENT) TYPE_T test[Lanes];

        ns::store(t, test, ns::max(t, ns::set(t, TYPE_T(10)), ns::set(t, TYPE_T(20))));
        EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(20)));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Max(-Inf, -100) = -100
            ns::store(t, test, ns::max(t, ns::set(t, -inf<TYPE_T>), ns::set(t, TYPE_T(-100))));
            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));

            // NaN 行为 (不同指令集行为不同，不进行测试)
            // ns::store(t, test, ns::max(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(-100))));
            // EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(-100)));

            // ns::store(t, test, ns::max(t, ns::set(t, TYPE_T(-100)), ns::set(t, qNaN<TYPE_T>)));
            // for (size_t i = 0; i < Lanes; ++i)
            // {
            //     EXPECT_TRUE(std::isnan(test[i]));
            // }

            // Check 模式，无论左右，都返回NaN
            ns::store(t, test, ns::max<ns::FloatMinMaxOption::CheckNaN>(t, ns::set(t, TYPE_T(5)), ns::set(t, qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(std::isnan(test[i]));
            }
            ns::store(t, test, ns::max<ns::FloatMinMaxOption::CheckNaN>(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, TYPE_T(5))));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 1 == 1 (True)
        auto mask = ns::reduce_mask(t, ns::equal(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(1))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));

        // 1 == 2 (False)
        mask = ns::reduce_mask(t, ns::equal(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(2))));
        EXPECT_TRUE(first_n_bit_false(mask, Lanes));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN == NaN (False)
            mask = ns::reduce_mask(t, ns::equal(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, qNaN<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_false(mask, Lanes));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void not_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 1 != 2 (True)
        auto mask = ns::reduce_mask(t, ns::not_equal(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(2))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // NaN != NaN (True)
            mask = ns::reduce_mask(t, ns::not_equal(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, qNaN<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_true(mask, Lanes));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 2 > 1 (True), 1 > 2 (False)
        auto mask = ns::reduce_mask(t, ns::greater(t, ns::set(t, TYPE_T(2)), ns::set(t, TYPE_T(1))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));

        mask = ns::reduce_mask(t, ns::greater(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(2))));
        EXPECT_TRUE(first_n_bit_false(mask, Lanes));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // Inf > 1e30 (True)
            mask = ns::reduce_mask(t, ns::greater(t, ns::set(t, inf<TYPE_T>), ns::set(t, TYPE_T(1e30))));
            EXPECT_TRUE(first_n_bit_true(mask, Lanes));

            // 1 > -Inf (True)
            mask = ns::reduce_mask(t, ns::greater(t, ns::set(t, TYPE_T(1)), ns::set(t, -inf<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_true(mask, Lanes));

            // NaN > 任何数 (False)
            mask = ns::reduce_mask(t, ns::greater(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, inf<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_false(mask, Lanes));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void greater_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 2 >= 2 (True)
        auto mask = ns::reduce_mask(t, ns::greater_equal(t, ns::set(t, TYPE_T(2)), ns::set(t, TYPE_T(2))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));
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
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 1 < 2 (True)
        auto mask = ns::reduce_mask(t, ns::less(t, ns::set(t, TYPE_T(1)), ns::set(t, TYPE_T(2))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));

        if constexpr (ksimd::is_scalar_floating_point<TYPE_T>) {
            // -Inf < Inf (True)
            mask = ns::reduce_mask(t, ns::less(t, ns::set(t, -inf<TYPE_T>), ns::set(t, inf<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_true(mask, Lanes));

            // NaN 相关比较应为 False
            mask = ns::reduce_mask(t, ns::less(t, ns::set(t, -inf<TYPE_T>), ns::set(t, qNaN<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_false(mask, Lanes));

            mask = ns::reduce_mask(t, ns::less(t, ns::set(t, qNaN<TYPE_T>), ns::set(t, -inf<TYPE_T>)));
            EXPECT_TRUE(first_n_bit_false(mask, Lanes));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void less_equal() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        const size_t Lanes = ns::lanes(t);
        
        // 5 <= 5 (True)
        auto mask = ns::reduce_mask(t, ns::less_equal(t, ns::set(t, TYPE_T(5)), ns::set(t, TYPE_T(5))));
        EXPECT_TRUE(first_n_bit_true(mask, Lanes));
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
    void mask_logic(size_t index) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        TAG_T t;

        const auto Lanes = ns::lanes(t);

        using mask_t = ns::Mask<decltype(t)>;
        using bitset_t = ns::MaskBitset<decltype(t)>;

        // 准备测试数据
        ns::Batch<decltype(t)> v1 = ns::set(t, static_cast<TYPE_T>(10));
        ns::Batch<decltype(t)> v2 = ns::set(t, static_cast<TYPE_T>(20));

        mask_t m_true  = ns::equal(t, v1, v1);  // All Ones
        mask_t m_false = ns::equal(t, v1, v2);  // All Zeros

        // 1. 基础位运算函数测试 (and, or, xor, not, and_not)
        {
            // AND: true & false == false
            EXPECT_TRUE(first_n_bit_false(ns::reduce_mask(t, ns::mask_and(t, m_true, m_false)), Lanes));

            // OR: true | false == true
            EXPECT_TRUE(first_n_bit_true(ns::reduce_mask(t, ns::mask_or(t, m_true, m_false)), Lanes));

            // XOR: true ^ true == false
            EXPECT_TRUE(first_n_bit_false(ns::reduce_mask(t, ns::mask_xor(t, m_true, m_true)), Lanes));

            // NOT: !true == false
            EXPECT_TRUE(first_n_bit_false(ns::reduce_mask(t, ns::mask_not(t, m_true)), Lanes));

            // NOT: !false == true
            EXPECT_TRUE(first_n_bit_true(ns::reduce_mask(t, ns::mask_not(t, m_false)), Lanes));

            // AND NOT: !false & true == true
            EXPECT_TRUE(first_n_bit_true(ns::reduce_mask(t, ns::mask_and_not(t, m_false, m_true)), Lanes));
        }

        // 2. 混合模式测试 (逐位验证)
        {
            // 手动构造一个混合数组，例如 [true, false, true, false]
            alignas(ALIGNMENT) TYPE_T data[ns::lanes(t)];
            for (size_t i = 0; i < ns::lanes(t); ++i) {
                data[i] = (i % 2 == 0) ? TYPE_T(10) : TYPE_T(20);
            }

            mask_t m_mixed = ns::equal(t, ns::load(t, data), v1);
            bitset_t res_mixed = ns::reduce_mask(t, m_mixed);

            bitset_t expected_mixed = 0;
            for (size_t i = 0; i < ns::lanes(t); i += 2) {
                expected_mixed |= (bitset_t(1) << i);
            }

            EXPECT_EQ(res_mixed, expected_mixed) << "idx: " << index;
        }
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN_WITH_IDX(mask_logic)
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

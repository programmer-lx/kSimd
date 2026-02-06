// using ALL_TYPE_T = uint32_t;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/base_op.hpp>

using namespace ksimd;

// ------------------------------------------ undefined ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void undefined() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        [[maybe_unused]] op::batch_t z = op::undefined();
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T arr[Lanes]{};

        op::batch_t z = op::zero();
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test_val[Lanes];

        // 测试常规数值广播
        TYPE_T val = TYPE_T(42);
        op::store(test_val, op::set(val));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test_val[i], val);

        // 针对浮点数的特殊值测试
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // 测试 NaN 广播
            op::store(test_val, op::set(qNaN<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test_val[i]));

            // 测试 Inf 广播
            op::store(test_val, op::set(inf<TYPE_T>));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test_val[i]));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test_val[Lanes];

        // 1. 测试无参 sequence(): [0, 1, 2, ...]
        op::store(test_val, op::sequence());
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test_val[i], static_cast<TYPE_T>(i));
        }

        // 2. 测试带 base 的 sequence(base): [base, base + 1, ...]
        TYPE_T base = TYPE_T(10);
        op::store(test_val, op::sequence(base));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test_val[i], static_cast<TYPE_T>(base + static_cast<TYPE_T>(i)));
        }

        // 3. 测试带 base 和 stride 的 sequence(base, stride): [base, base + stride, ...]
        TYPE_T base_v = TYPE_T(5);
        TYPE_T stride = TYPE_T(2);
        op::store(test_val, op::sequence(base_v, stride));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test_val[i], static_cast<TYPE_T>(base_v + static_cast<TYPE_T>(i) * stride));
        }

        // 4. 针对浮点数的特殊测试（如负步长或小数步长）
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            TYPE_T f_base = TYPE_T(1.5);
            TYPE_T f_stride = TYPE_T(-0.5);
            op::store(test_val, op::sequence(f_base, f_stride));
            for (size_t i = 0; i < Lanes; ++i) {
                // 使用预期值进行比较，浮点数在此类简单加法中通常是精确的
                EXPECT_NEAR(test_val[i], f_base + static_cast<TYPE_T>(i) * f_stride, TYPE_T(1e-6));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T in[Lanes];
        alignas(ALIGNMENT) TYPE_T out[Lanes];

        // 初始化数据
        for (size_t i = 0; i < Lanes; ++i) {
            in[i] = TYPE_T(i + 7);
            out[i] = TYPE_T(0);
        }

        // 执行对齐的读写
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        // 分配比 Lanes 稍大的空间，用于模拟非对齐起始地址
        alignas(ALIGNMENT) TYPE_T buffer_in[Lanes + 1];
        alignas(ALIGNMENT) TYPE_T buffer_out[Lanes + 1];

        // 使用偏移量 1 来确保地址不再满足 ALIGNMENT 对齐要求
        TYPE_T* unaligned_in = buffer_in + 1;
        TYPE_T* unaligned_out = buffer_out + 1;

        for (size_t i = 0; i < Lanes; ++i) {
            unaligned_in[i] = TYPE_T(i * 3 + 1);
            unaligned_out[i] = TYPE_T(0);
        }

        // 执行非对齐的读写
        op::storeu(unaligned_out, op::loadu(unaligned_in));

        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(unaligned_out[i], unaligned_in[i]);
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(loadu_storeu)
#endif

// ------------------------------------------ bit_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_select() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using uint_t = same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T res[Lanes];

        // 严格遵循你的实现：(mask & a) | (~mask & b)
        // a:    0b10101 (0x15)
        // b:    0b11111 (0x1F)
        // mask: 0b00010 (0x02)
        // -----------------------
        // mask & a   => 0b00010 & 0b10101 = 0b00000
        // ~mask & b  => 0b11101 & 0b11111 = 0b11101 (在5bit下)
        // 结果: 0b00000 | 0b11101 = 0b11101 (0x1D)

        // 我们直接用你提供的测试数据进行验证：
        // a = 0b10101, b = 0b11111, mask = 0b00010
        // 预期结果 = 0b10111 (这是你最开始代码里给出的预期)
        // 让我们手动算一下你的源码：
        // (0b00010 & 0b10101) | (~0b00010 & 0b11111)
        // = (0b00000) | (0b...11101 & 0b11111) = 0b11101

        // 为了避免歧义，我们用一组最简单的位模式：
        TYPE_T val_a    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b10101));
        TYPE_T val_b    = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b11111));
        TYPE_T val_mask = make_var_from_bits<TYPE_T>(static_cast<uint_t>(0b00010));

        // 按照 (mask & a) | (~mask & b) 计算：
        uint_t expected = static_cast<uint_t>(0b11101);

        op::store(res, op::bit_select(op::set(val_mask), op::set(val_a), op::set(val_b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(array_bit_equal(&res[i], 1, expected))
                << "Bit select failed at lane " << i
                << "\n  Expected: 0x" << std::hex << (uint64_t)expected
                << "\n  Actual:   0x" << (uint64_t)std::bit_cast<uint_t>(res[i]);
        }

        // 浮点数简单验证：符号位搬运
        if constexpr (std::is_floating_point_v<TYPE_T>)
        {
            TYPE_T positive = TYPE_T(1.0);
            TYPE_T negative = TYPE_T(-2.0);
            TYPE_T mask = ksimd::SignBitMask<TYPE_T>; // 符号位为1

            // mask为1选a(positive)，结果应为 +2.0 (因为数值位全从b拿，b是~mask)
            // 这个逻辑在浮点数上比较绕，通常 bit_select(mask, a, b)
            // 如果想实现 copysign，mask 应该是数值位的掩码。
            // 我们仅验证位逻辑正确即可。
            op::store(res, op::bit_select(op::set(mask), op::set(negative), op::set(positive)));

            // 符号位mask是1，选了negative的符号位(1)
            // 其他位mask是0，选了positive的数值位
            // 结果应该是 -1.0
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(res[i], TYPE_T(-1.0));
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(bit_select)
#endif

// ------------------------------------------ mask_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_select() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 构造两个基础向量
        auto vec_a = op::set(TYPE_T(10));
        auto vec_b = op::set(TYPE_T(20));

        // Case 1: 全 1 掩码 (应该全部选 a)
        {
            FILL_ARRAY(test, -1);
            auto mask = op::equal(op::set(1), op::set(1)); // 产生全 1 掩码
            op::store(test, op::mask_select(mask, vec_a, vec_b));

            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(10)));
        }

        // Case 2: 全 0 掩码 (应该全部选 b)
        {
            FILL_ARRAY(test, -1);
            auto mask = op::equal(op::set(1), op::set(2)); // 产生全 0 掩码
            op::store(test, op::mask_select(mask, vec_a, vec_b));

            EXPECT_TRUE(array_equal(test, Lanes, TYPE_T(20)));
        }

        // Case 3: 混合掩码 (交叉选择)
        // 注意：这里假设你有 op::load 或类似方式构造非齐次向量进行测试
        // 如果为了简单，可以直接利用比较指令产生交叉掩码
        {
            alignas(ALIGNMENT) TYPE_T data_i[Lanes];
            alignas(ALIGNMENT) TYPE_T data_threshold[Lanes];
            for(size_t i = 0; i < Lanes; ++i) {
                data_i[i] = (TYPE_T)i;
                data_threshold[i] = TYPE_T(1);
            }

            FILL_ARRAY(test, TYPE_T(-1));
            auto mask = op::greater(op::load(data_i), op::load(data_threshold)); 
            // mask 为 [F, F, T, T, ...] (取决于 Lanes 长度)
            
            op::store(test, op::mask_select(mask, vec_a, vec_b));

            for(size_t i = 0; i < Lanes; ++i) {
                TYPE_T expected_val = (i > 1) ? TYPE_T(10) : TYPE_T(20);
                EXPECT_EQ(test[i], expected_val);
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(mask_select)
#endif

// ------------------------------------------ bit_not ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_not() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 构造输入数据: ...010101 (21)
        // 期望结果 (NOT): ...101010
        // 我们只验证低 5 位
        TYPE_T input_val = make_var_from_bits<TYPE_T>(0b10101);

        FILL_ARRAY(test, -1);
        op::store(test, op::bit_not(op::set(input_val)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            auto result_bits = ksimd::bitcast_to_uint(test[i]);

            // bit 0: 1 -> 0 (False)
            EXPECT_FALSE(test_bit(result_bits, 0));
            // bit 1: 0 -> 1 (True)
            EXPECT_TRUE(test_bit(result_bits, 1));
            // bit 2: 1 -> 0 (False)
            EXPECT_FALSE(test_bit(result_bits, 2));
            // bit 3: 0 -> 1 (True)
            EXPECT_TRUE(test_bit(result_bits, 3));
            // bit 4: 1 -> 0 (False)
            EXPECT_FALSE(test_bit(result_bits, 4));
        }

        // 额外的全 0 测试 -> 应变为全 1 (ksimd::one_block 的位模式)
        {
            FILL_ARRAY(test, -1);
            op::store(test, op::bit_not(op::set(TYPE_T(0))));
            // 对于 IEEE 754 浮点数，全 0 取反不等于 one_block，
            // 但位模式应该是全 F。这里直接验证位 cast 后的结果。
            for (size_t i = 0; i < Lanes; ++i)
            {
                auto result_bits = ksimd::bitcast_to_uint(test[i]);
                EXPECT_EQ(result_bits, ~ksimd::bitcast_to_uint(TYPE_T(0)));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // a: ...10101
        // b: ...10011
        // r: ...10001
        TYPE_T a = make_var_from_bits<TYPE_T>(0b10101);
        TYPE_T b = make_var_from_bits<TYPE_T>(0b10011);
        TYPE_T expected = make_var_from_bits<TYPE_T>(0b10001);

        FILL_ARRAY(test, -1);
        op::store(test, op::bit_and(op::set(a), op::set(b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(ksimd::bitcast_to_uint(test[i]), ksimd::bitcast_to_uint(expected));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // a:     ...10101 -> NOT a: ...01010
        // b:     ...10011
        // result: ...00010 (AND)
        TYPE_T a = make_var_from_bits<TYPE_T>(0b10101);
        TYPE_T b = make_var_from_bits<TYPE_T>(0b10011);
        TYPE_T expected = make_var_from_bits<TYPE_T>(0b00010);

        FILL_ARRAY(test, -1);
        op::store(test, op::bit_and_not(op::set(a), op::set(b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(ksimd::bitcast_to_uint(test[i]), ksimd::bitcast_to_uint(expected));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // a: ...10101
        // b: ...10011
        // r: ...10111
        TYPE_T a = make_var_from_bits<TYPE_T>(0b10101);
        TYPE_T b = make_var_from_bits<TYPE_T>(0b10011);
        TYPE_T expected = make_var_from_bits<TYPE_T>(0b10111);

        FILL_ARRAY(test, -1);
        op::store(test, op::bit_or(op::set(a), op::set(b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(ksimd::bitcast_to_uint(test[i]), ksimd::bitcast_to_uint(expected));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // a: ...10101
        // b: ...10011
        // r: ...00110
        TYPE_T a = make_var_from_bits<TYPE_T>(0b10101);
        TYPE_T b = make_var_from_bits<TYPE_T>(0b10011);
        TYPE_T expected = make_var_from_bits<TYPE_T>(0b00110);

        FILL_ARRAY(test, -1);
        op::store(test, op::bit_xor(op::set(a), op::set(b)));

        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_EQ(ksimd::bitcast_to_uint(test[i]), ksimd::bitcast_to_uint(expected));
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 常规数值测试
        op::store(test, op::div(op::set(TYPE_T(100)), op::set(TYPE_T(4))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_NEAR(test[i], TYPE_T(25), std::numeric_limits<TYPE_T>::epsilon());

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

// ------------------------------------------ all_operators ------------------------------------------
#define EXPECT_BATCH_BIT_EQ(actual, expected, op_name)                     \
{                                                                          \
    alignas(ALIGNMENT) TYPE_T act_buf[Lanes];                              \
    alignas(ALIGNMENT) TYPE_T exp_buf[Lanes];                              \
    op::store(act_buf, actual);                                            \
    op::store(exp_buf, expected);                                          \
    for (size_t i = 0; i < Lanes; ++i) {                                   \
        EXPECT_TRUE(bit_equal(act_buf[i], std::bit_cast<int_t>(exp_buf[i])))\
            << "Operator [" << op_name << "] failed at lane " << i;        \
    }                                                                      \
}
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void all_operators() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using batch_t = typename op::batch_t;
        using int_t = same_bits_uint_t<TYPE_T>; // 对应浮点大小的整数类型
        constexpr size_t Lanes = op::TotalLanes;

        // 准备基础数据
        batch_t a = op::set(TYPE_T(100));
        batch_t b = op::set(TYPE_T(2));
        batch_t zero = op::set(TYPE_T(0));

        // 1. 二元算术运算符测试 (Binary Arithmetic)
        EXPECT_BATCH_BIT_EQ(a + b, op::add(a, b), "+");
        EXPECT_BATCH_BIT_EQ(a - b, op::sub(a, b), "-");
        EXPECT_BATCH_BIT_EQ(a * b, op::mul(a, b), "*");
        EXPECT_BATCH_BIT_EQ(a / b, op::div(a, b), "/");

        // 2. 一元运算符测试 (Unary)
        EXPECT_BATCH_BIT_EQ(-a, op::sub(zero, a), "unary -");

        // 3. 位逻辑运算符测试 (Bitwise)
        // 使用特殊的位模式：0x55 (0101) 和 0xAA (1010)
        batch_t m1 = op::set(make_var_from_bits<TYPE_T>(static_cast<int_t>(0x5555555555555555ULL)));
        batch_t m2 = op::set(make_var_from_bits<TYPE_T>(static_cast<int_t>(0xAAAAAAAAAAAAAAAAULL)));

        EXPECT_BATCH_BIT_EQ(m1 & m2, op::bit_and(m1, m2), "&");
        EXPECT_BATCH_BIT_EQ(m1 | m2, op::bit_or(m1, m2), "|");
        EXPECT_BATCH_BIT_EQ(m1 ^ m2, op::bit_xor(m1, m2), "^");
        EXPECT_BATCH_BIT_EQ(~m1,     op::bit_not(m1), "~");

        // 4. 复合赋值运算符测试 (Compound Assignment)
        batch_t c = a;
        c += b; EXPECT_BATCH_BIT_EQ(c, op::add(a, b), "+=");
        c = a;
        c -= b; EXPECT_BATCH_BIT_EQ(c, op::sub(a, b), "-=");
        c = a;
        c *= b; EXPECT_BATCH_BIT_EQ(c, op::mul(a, b), "*=");
        c = a;
        c /= b; EXPECT_BATCH_BIT_EQ(c, op::div(a, b), "/=");

        c = m1;
        c &= m2; EXPECT_BATCH_BIT_EQ(c, op::bit_and(m1, m2), "&=");
        c = m1;
        c |= m2; EXPECT_BATCH_BIT_EQ(c, op::bit_or(m1, m2), "|=");
        c = m1;
        c ^= m2; EXPECT_BATCH_BIT_EQ(c, op::bit_xor(m1, m2), "^=");

        // 5. 浮点极端边界测试 (IEEE 754 Consistency)
        if constexpr (std::is_floating_point_v<TYPE_T>) {
            batch_t pinf = op::set(inf<TYPE_T>);
            batch_t nan  = op::set(qNaN<TYPE_T>);

            // Inf * 2.0 = Inf
            EXPECT_BATCH_BIT_EQ(pinf * op::set(TYPE_T(2)), pinf, "inf * 2");
            // Inf - Inf = NaN (验证运算符产生 NaN 的行为与指令一致)
            EXPECT_BATCH_BIT_EQ(pinf - pinf, op::sub(pinf, pinf), "inf - inf");
            // NaN 逻辑传播
            EXPECT_BATCH_BIT_EQ(nan + a, op::add(nan, a), "nan + a");
            // 位操作：~0.0f
            EXPECT_BATCH_BIT_EQ(~zero, op::bit_not(zero), "~zero");
        }
    }
}

#undef EXPECT_BATCH_BIT_EQ

#if KSIMD_ONCE
TEST_ONCE_DYN(all_operators)
#endif

// ------------------------------------------ reduce_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_add() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;

        alignas(ALIGNMENT) TYPE_T data[Lanes];
        TYPE_T expected = 0;
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
            expected += data[i];
        }

        TYPE_T res = op::reduce_add(op::load(data));
        EXPECT_NEAR(res, expected, std::numeric_limits<TYPE_T>::epsilon());

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // (2 * 3) + 4 = 10
        op::store(test, op::mul_add(op::set(TYPE_T(2)), op::set(TYPE_T(3)), op::set(TYPE_T(4))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(10));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN propagation: (NaN * 2) + 3 = NaN
            op::store(test, op::mul_add(op::set(qNaN<TYPE_T>), op::set(TYPE_T(2)), op::set(TYPE_T(3))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));

            // Inf propagation: (Inf * 2) + 3 = Inf
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        op::store(test, op::min(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(10));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Min(Inf, 100) = 100
            op::store(test, op::min(op::set(inf<TYPE_T>), op::set(TYPE_T(100))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(100));

            // Min(100, Inf) = 100
            op::store(test, op::min(op::set(TYPE_T(100)), op::set(inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(100));

            // NaN 行为
            op::store(test, op::min(op::set(qNaN<TYPE_T>), op::set(TYPE_T(5))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(5));

            op::store(test, op::min(op::set(TYPE_T(5)), op::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(min)
#endif


// ------------------------------------------ max ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void max() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        op::store(test, op::max(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(20));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Max(-Inf, -100) = -100
            op::store(test, op::max(op::set(-inf<TYPE_T>), op::set(TYPE_T(-100))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(-100));

            // NaN
            op::store(test, op::max(op::set(qNaN<TYPE_T>), op::set(TYPE_T(-100))));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(-100));

            op::store(test, op::max(op::set(TYPE_T(-100)), op::set(qNaN<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isnan(test[i]));
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(max)
#endif

// ------------------------------------------ equal ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void equal() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 基础测试: 1 == 1 (True), 1 == 2 (False)
        op::test_store_mask(test, op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(1))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        op::test_store_mask(test, op::equal(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Inf == Inf (True)
            op::test_store_mask(test, op::equal(op::set(inf<TYPE_T>), op::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN == NaN (False)
            op::test_store_mask(test, op::equal(op::set(qNaN<TYPE_T>), op::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void not_equal() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 基础测试: 1 != 2 (True)
        op::test_store_mask(test, op::not_equal(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // NaN != NaN (True) - IEEE 754 核心准则
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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

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
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 1 < 2 (True)
        op::test_store_mask(test, op::less(op::set(TYPE_T(1)), op::set(TYPE_T(2))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // -Inf < Inf (True)
            op::test_store_mask(test, op::less(op::set(-inf<TYPE_T>), op::set(inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));

            // NaN (False)
            op::test_store_mask(test, op::less(op::set(-inf<TYPE_T>), op::set(qNaN<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));

            op::test_store_mask(test, op::less(op::set(qNaN<TYPE_T>), op::set(-inf<TYPE_T>)));
            EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::ZeroBlock<TYPE_T>));
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void less_equal() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::TotalLanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 5 <= 5 (True)
        op::test_store_mask(test, op::less_equal(op::set(TYPE_T(5)), op::set(TYPE_T(5))));
        EXPECT_TRUE(array_bit_equal(test, Lanes, ksimd::OneBlock<TYPE_T>));
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(less)
TEST_ONCE_DYN(less_equal)
#endif


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
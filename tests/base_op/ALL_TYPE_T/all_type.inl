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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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

// ------------------------------------------ load_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void load_store() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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

// ------------------------------------------ mask_load_mask_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_load_mask_store() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using mask_t = typename op::mask_t;
        using batch_t = typename op::batch_t;
        constexpr size_t Lanes = op::Lanes;

        // 准备源数据和测试缓冲区
        alignas(ALIGNMENT) TYPE_T src[Lanes];
        alignas(ALIGNMENT) TYPE_T dst[Lanes];
        for (size_t i = 0; i < Lanes; ++i) src[i] = TYPE_T(i + 1); // 1, 2, 3...

        // 1. 全掩码测试 (Full Mask)
        {
            FILL_ARRAY(dst, TYPE_T(-1));
            mask_t mask = op::mask_from_lanes(Lanes);
            batch_t data = op::mask_load(src, mask);
            op::mask_store(dst, data, mask);

            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_TRUE(dst[i] == src[i]);
            }
        }

        // 2. 零掩码测试 (Zero Mask)
        {
            FILL_ARRAY(dst, TYPE_T(77));
            mask_t mask = op::mask_from_lanes(0);
            batch_t data = op::mask_load(src, mask);

            // 验证 load 结果是否为 0 (Zeroing 语义)
            alignas(ALIGNMENT) TYPE_T load_res[Lanes];
            op::store(load_res, data);
            EXPECT_TRUE(array_bit_equal(load_res, Lanes, ksimd::ZeroBlock<TYPE_T>));

            // 验证 store 是否不写入任何内容
            op::mask_store(dst, data, mask);
            for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(dst[i], TYPE_T(77));
        }

        // 3. 边界测试：加载前 N-1 个元素 (Partial Tail)
        {
            FILL_ARRAY(dst, TYPE_T(0));
            mask_t mask = op::mask_from_lanes(Lanes - 1);
            batch_t data = op::mask_load(src, mask);
            op::mask_store(dst, data, mask);

            for (size_t i = 0; i < Lanes; ++i)
            {
                if (i < Lanes - 1) {
                    EXPECT_EQ(dst[i], src[i]);
                } else {
                    EXPECT_EQ(dst[i], TYPE_T(0));
                }
            }
        }

        // 4. 任意位置掩码测试 (Arbitrary Mask)
        // 只有偶数下标被激活
        {
            FILL_ARRAY(dst, TYPE_T(88));
            // 假设你有通过比较产生 mask 的通用方式
            // 或者通过 bit 构造（取决于你的 mask_t 实现）
            // 这里使用通用的 greater 来产生间隔掩码作为示例
            alignas(ALIGNMENT) TYPE_T pattern[Lanes];
            alignas(ALIGNMENT) TYPE_T threshold[Lanes];
            for(size_t i=0; i<Lanes; ++i) {
                pattern[i] = TYPE_T(i);
                threshold[i] = TYPE_T(Lanes / 2); // 只是为了产生一种混合状态
            }
            mask_t mask = op::greater(op::load(pattern), op::load(threshold));

            batch_t data = op::mask_load(src, mask);
            op::mask_store(dst, data, mask);

            alignas(ALIGNMENT) TYPE_T src_loaded[Lanes];
            op::store(src_loaded, data);

            for (size_t i = 0; i < Lanes; ++i)
            {
                // 手动模拟逻辑
                if (i > (Lanes / 2)) {
                    EXPECT_EQ(dst[i], src[i]);
                    EXPECT_EQ(src_loaded[i], src[i]);
                } else {
                    EXPECT_EQ(dst[i], TYPE_T(88)); // store 没发生
                    EXPECT_EQ(src_loaded[i], TYPE_T(0)); // load 填了 0
                }
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(mask_load_mask_store)
#endif

// ------------------------------------------ mask_loadu_mask_storeu ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_loadu_mask_storeu() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using mask_t = typename op::mask_t;
        using batch_t = typename op::batch_t;
        constexpr size_t Lanes = op::Lanes;

        // 为了测试真正的非对齐（Unaligned），我们分配比 Lanes 大的空间
        // 并从 offset=1 的地方开始读写
        constexpr size_t BufferSize = Lanes + 1;
        alignas(ALIGNMENT) TYPE_T src_raw[BufferSize];
        alignas(ALIGNMENT) TYPE_T dst_raw[BufferSize];

        for (size_t i = 0; i < BufferSize; ++i) {
            src_raw[i] = TYPE_T(i + 1);
            dst_raw[i] = TYPE_T(0);
        }

        // 强制获取非对齐指针
        const TYPE_T* unaligned_src = src_raw + 1;
        TYPE_T* unaligned_dst = dst_raw + 1;

        // 1. 全掩码非对齐测试 (Full Mask, Unaligned)
        {
            FILL_ARRAY(dst_raw, TYPE_T(0));
            mask_t mask = op::mask_from_lanes(Lanes);

            // 执行非对齐加载
            batch_t data = op::mask_loadu(unaligned_src, mask);
            // 执行非对齐存储
            op::mask_storeu(unaligned_dst, data, mask);

            // 验证：dst_raw[0] 应该是 0（未被触碰），dst_raw[1...Lanes] 应该是 src_raw[1...Lanes]
            EXPECT_EQ(dst_raw[0], TYPE_T(0));
            for (size_t i = 0; i < Lanes; ++i)
            {
                EXPECT_EQ(unaligned_dst[i], unaligned_src[i]);
            }
        }

        // 2. 尾部部分加载测试 (Partial Tail, Unaligned)
        // 模拟处理数组最后几个元素的情况
        {
            FILL_ARRAY(dst_raw, TYPE_T(77));
            constexpr size_t PartialCount = Lanes / 2;
            mask_t mask = op::mask_from_lanes(PartialCount);

            batch_t data = op::mask_loadu(unaligned_src, mask);
            op::mask_storeu(unaligned_dst, data, mask);

            // 验证 Zeroing 语义：Load 进来的数据，掩码外应该是 0
            alignas(ALIGNMENT) TYPE_T internal[Lanes];
            op::store(internal, data);

            for (size_t i = 0; i < Lanes; ++i)
            {
                if (i < PartialCount) {
                    EXPECT_EQ(internal[i], unaligned_src[i]);
                    EXPECT_EQ(unaligned_dst[i], unaligned_src[i]);
                } else {
                    EXPECT_EQ(internal[i], TYPE_T(0));
                    EXPECT_EQ(unaligned_dst[i], TYPE_T(77)); // Store 应该受保护，不覆盖原有的 77
                }
            }
        }

        // 3. 跨页/非对齐安全性验证 (Arbitrary Mask)
        {
            // 构造一个只有首尾被激活的掩码
            alignas(ALIGNMENT) TYPE_T mask_arr[Lanes];
            memset(mask_arr, 0x00, sizeof(mask_arr));
            mask_arr[0] = OneBlock<TYPE_T>;
            mask_arr[Lanes - 1] = OneBlock<TYPE_T>;
            mask_t mask = op::test_load_mask(mask_arr);

            FILL_ARRAY(dst_raw, TYPE_T(99));
            batch_t data = op::mask_loadu(unaligned_src, mask);
            op::mask_storeu(unaligned_dst, data, mask);

            for (size_t i = 0; i < Lanes; ++i)
            {
                if (i == 0 || i == (Lanes - 1)) {
                    EXPECT_EQ(unaligned_dst[i], unaligned_src[i]);
                } else {
                    EXPECT_EQ(unaligned_dst[i], TYPE_T(99)); // 保持不变
                }
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(mask_loadu_mask_storeu)
#endif

// ------------------------------------------ mask_from_lanes ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_from_lanes() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using mask_t = op::mask_t;
        constexpr size_t lanes = op::Lanes;

        alignas(ALIGNMENT) TYPE_T arr[lanes] = {};
        alignas(ALIGNMENT) TYPE_T dst[lanes]{};

        // [0, lane - 1] == 1
        mask_t mask = op::mask_from_lanes(lanes - 1);
        FILL_ARRAY(arr, OneBlock<TYPE_T>);
        arr[lanes - 1] = TYPE_T(0);
        FILL_ARRAY(dst, TYPE_T(100));
        op::test_store_mask(dst, mask);
        for (size_t i = 0; i< lanes; ++i)
            EXPECT_TRUE(bit_equal(dst, arr));

        // [all] == 1
        mask = op::mask_from_lanes(1000);
        FILL_ARRAY(arr, OneBlock<TYPE_T>);
        FILL_ARRAY(dst, TYPE_T(100));
        op::test_store_mask(dst, mask);
        for (size_t i = 0; i< lanes; ++i)
            EXPECT_TRUE(bit_equal(dst, arr));

        // [all] == 0
        mask = op::mask_from_lanes(0);
        FILL_ARRAY(arr, ZeroBlock<TYPE_T>);
        FILL_ARRAY(dst, TYPE_T(100));
        op::test_store_mask(dst, mask);
        for (size_t i = 0; i< lanes; ++i)
            EXPECT_TRUE(bit_equal(dst, arr));
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(mask_from_lanes);

TEST(dyn_dispatch_TYPE_T, mask_from_lanes)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_from_lanes)); ++idx)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_from_lanes)[idx]();
    }
}
#endif

// ------------------------------------------ bit_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void bit_select() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        using uint_t = same_bits_uint_t<TYPE_T>;
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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
        constexpr size_t Lanes = op::Lanes;

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

// ------------------------------------------ reduce_sum ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void reduce_sum() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::Lanes;

        alignas(ALIGNMENT) TYPE_T data[Lanes];
        TYPE_T expected = 0;
        for (size_t i = 0; i < Lanes; ++i) {
            data[i] = TYPE_T(i + 1);
            expected += data[i];
        }

        TYPE_T res = op::reduce_sum(op::load(data));
        EXPECT_NEAR(res, expected, std::numeric_limits<TYPE_T>::epsilon());

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Inf in sum
            data[0] = inf<TYPE_T>;
            EXPECT_TRUE(std::isinf(op::reduce_sum(op::load(data))));

            // NaN in sum
            data[0] = qNaN<TYPE_T>;
            EXPECT_TRUE(std::isnan(op::reduce_sum(op::load(data))));
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(reduce_sum)
#endif

// ------------------------------------------ mul_add ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mul_add() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::Lanes;
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

// ------------------------------------------ abs ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void abs() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // 正数与负数
        op::store(test, op::abs(op::set(TYPE_T(-5))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(5));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // -0.0 -> 0.0
            op::store(test, op::abs(op::set(TYPE_T(-0.0))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(test[i], TYPE_T(0.0));
                EXPECT_FALSE(std::signbit(test[i])); // 验证符号位已清除
            }

            // -Inf -> Inf
            op::store(test, op::abs(op::set(-inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) EXPECT_TRUE(std::isinf(test[i]) && test[i] > 0);
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(abs)
#endif

// ------------------------------------------ min ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void min() noexcept
    {
        using op = KSIMD_DYN_BASE_OP(TYPE_T);
        constexpr size_t Lanes = op::Lanes;
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        op::store(test, op::min(op::set(TYPE_T(10)), op::set(TYPE_T(20))));
        for (size_t i = 0; i < Lanes; ++i) EXPECT_EQ(test[i], TYPE_T(10));

        if constexpr (std::is_floating_point_v<TYPE_T>) {
            // Min(Inf, 100) = 100
            op::store(test, op::min(op::set(inf<TYPE_T>), op::set(TYPE_T(100))));
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
        constexpr size_t Lanes = op::Lanes;
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
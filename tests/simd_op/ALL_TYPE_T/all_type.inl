// using ALL_TYPE_T = uint32_t;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/simd_op.hpp>

using namespace ksimd;

// ------------------------------------------ undefined ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void undefined() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
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
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
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
    void set(TYPE_T x, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::set(x));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(set);

TEST(dyn_dispatch_TYPE_T, set)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) out[i] = static_cast<TYPE_T>(123);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)[idx](static_cast<TYPE_T>(3), out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == static_cast<TYPE_T>(3));
    }
}
#endif

// ------------------------------------------ load + store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void load_store(const TYPE_T* KSIMD_RESTRICT in, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::store(out + i, op::load(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(load_store);

TEST(dyn_dispatch_TYPE_T, load_store)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = TYPE_T(i + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = TYPE_T(456);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == in[i]);
    }
}
#endif

// ------------------------------------------ loadu + storeu ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void loadu_storeu(const TYPE_T* KSIMD_RESTRICT in, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::loadu(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(loadu_storeu);

TEST(dyn_dispatch_TYPE_T, loadu_storeu)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = TYPE_T(i * 2 + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = TYPE_T(999);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == in[i]);
    }
}
#endif

// ------------------------------------------ mask_load_mask_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_load_mask_store() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
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
            EXPECT_TRUE(array_bit_equal(load_res, Lanes, ksimd::zero_block<TYPE_T>));

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
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
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
            mask_arr[0] = one_block<TYPE_T>;
            mask_arr[Lanes - 1] = one_block<TYPE_T>;
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
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        using mask_t = op::mask_t;
        constexpr size_t lanes = op::Lanes;

        alignas(ALIGNMENT) TYPE_T arr[lanes] = {};
        alignas(ALIGNMENT) TYPE_T dst[lanes]{};

        // [0, lane - 1] == 1
        mask_t mask = op::mask_from_lanes(lanes - 1);
        FILL_ARRAY(arr, one_block<TYPE_T>);
        arr[lanes - 1] = TYPE_T(0);
        FILL_ARRAY(dst, TYPE_T(100));
        op::test_store_mask(dst, mask);
        for (size_t i = 0; i< lanes; ++i)
            EXPECT_TRUE(bit_equal(dst, arr));

        // [all] == 1
        mask = op::mask_from_lanes(1000);
        FILL_ARRAY(arr, one_block<TYPE_T>);
        FILL_ARRAY(dst, TYPE_T(100));
        op::test_store_mask(dst, mask);
        for (size_t i = 0; i< lanes; ++i)
            EXPECT_TRUE(bit_equal(dst, arr));

        // [all] == 0
        mask = op::mask_from_lanes(0);
        FILL_ARRAY(arr, zero_block<TYPE_T>);
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
    void bit_select(
        const TYPE_T* a,
        const TYPE_T* b,
        const TYPE_T* c,
        TYPE_T* out) noexcept
    {
        
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            auto result = op::bit_select(op::load(a + i), op::load(b + i), op::load(c + i));
            op::store(out + i, result);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(bit_select);

TEST(dyn_dispatch_TYPE_T, bit_select)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_select)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T a[TOTAL];
        alignas(ALIGNMENT) TYPE_T b[TOTAL];
        alignas(ALIGNMENT) TYPE_T c[TOTAL];
        alignas(ALIGNMENT) TYPE_T test[TOTAL];

        FILL_ARRAY(a, make_float_from_bits<TYPE_T>(0b10101));
        FILL_ARRAY(b, make_float_from_bits<TYPE_T>(0b11111));
        FILL_ARRAY(c, make_float_from_bits<TYPE_T>(0b00010));
        FILL_ARRAY(test, -1);
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(bit_select)[idx](a, b, c, test);
        EXPECT_TRUE(array_bit_equal(test, std::size(test), make_float_from_bits<TYPE_T>(0b10111)));
    }
}
#endif

// ------------------------------------------ mask_select ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_select() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
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


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
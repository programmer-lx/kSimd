#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "packed_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/packed_op.hpp>

using namespace ksimd;

// ------------------------------------------ sequence ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sequence() noexcept
    {
        // 假设通过外部注入或上下文获取 TYPE_T 和具体的 WIDTH
        using op = KSIMD_DYN_PACKED_OP(TYPE_T, WIDTH);
        
        constexpr size_t Lanes = op::TotalLanes; // 总通道数 (e.g., 8)
        constexpr size_t VecWidth = op::Width;   // 单个向量宽度 (e.g., 4)
        constexpr size_t VecCount = op::Count;   // 向量个数 (e.g., 2)
        
        alignas(ALIGNMENT) TYPE_T test[Lanes]{};

        // --- 1. 基础测试: sequence() [默认步长1, 基数0, 周期性循环] ---
        op::store(test, op::sequence());
        for (size_t c = 0; c < VecCount; ++c) {
            for (size_t w = 0; w < VecWidth; ++w) {
                // 根据你的代码 _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0)
                // 预期每个 block 都是 0, 1, 2, 3...
                EXPECT_EQ(test[c * VecWidth + w], TYPE_T(w));
            }
        }

        // --- 2. 带基数测试: sequence(base) ---
        const TYPE_T base = TYPE_T(10);
        op::store(test, op::sequence(base));
        for (size_t c = 0; c < VecCount; ++c) {
            for (size_t w = 0; w < VecWidth; ++w) {
                EXPECT_EQ(test[c * VecWidth + w], base + TYPE_T(w));
            }
        }

        // --- 3. 带基数与步长测试: sequence(base, stride) ---
        const TYPE_T base2 = TYPE_T(5);
        const TYPE_T stride = TYPE_T(2);
        op::store(test, op::sequence(base2, stride));
        for (size_t c = 0; c < VecCount; ++c) {
            for (size_t w = 0; w < VecWidth; ++w) {
                // 预期: base + w * stride
                EXPECT_EQ(test[c * VecWidth + w], base2 + TYPE_T(w) * stride);
            }
        }

        // --- 4. 边界与特殊值测试 (极值步长) ---
        const TYPE_T big_base = std::numeric_limits<TYPE_T>::max() - TYPE_T(10);
        const TYPE_T zero_stride = TYPE_T(0);
        
        // 测试零步长: 应该退化为全 base 效果
        op::store(test, op::sequence(big_base, zero_stride));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(test[i], big_base);
        }

        // --- 5. 负向步长测试 ---
        if constexpr (std::is_signed_v<TYPE_T>)
        {
            const TYPE_T neg_stride = TYPE_T(-1.0);
            op::store(test, op::sequence(TYPE_T(10.0), neg_stride));
            for (size_t c = 0; c < VecCount; ++c) {
                for (size_t w = 0; w < VecWidth; ++w) {
                    EXPECT_EQ(test[c * VecWidth + w], TYPE_T(10.0) - TYPE_T(w));
                }
            }
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(sequence)
#endif

// ------------------------------------------ merge_width4 ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void merge_width4() noexcept
    {
        // 显式获取逻辑宽度为 4 的算子
        using op = KSIMD_DYN_PACKED_OP(TYPE_T, 4);

        // 逻辑属性
        constexpr size_t LCount = op::Count;   // 逻辑块数量 (对于 AVX float32 是 2)
        constexpr size_t LWidth = op::Width;   // 逻辑块宽度 (固定为 4)
        constexpr size_t TotalLanes = op::TotalLanes; // 逻辑总通道 (2 * 4 = 8)

        // 物理断言：确保在 AVX 下物理寄存器确实只用了 1 个，但支撑了 2 个逻辑块
        static_assert(LWidth == 4, "Logical width must be 4");
        static_assert(TotalLanes == LCount * LWidth, "Consistency check");

        alignas(ALIGNMENT) TYPE_T data_a[TotalLanes];
        alignas(ALIGNMENT) TYPE_T data_b[TotalLanes];
        alignas(ALIGNMENT) TYPE_T result[TotalLanes];

        // 1. 初始化数据：按照逻辑块填充
        // Block 0: [0, 1, 2, 3]   | Block 1: [4, 5, 6, 7]
        // Block 0: [10, 11, 12, 13] | Block 1: [14, 15, 16, 17]
        for (size_t c = 0; c < LCount; ++c) {
            for (size_t w = 0; w < LWidth; ++w) {
                size_t idx = c * LWidth + w;
                data_a[idx] = TYPE_T(idx);
                data_b[idx] = TYPE_T(idx + 10);
            }
        }

        auto batch_a = op::load(data_a);
        auto batch_b = op::load(data_b);

        // --- 执行测试: merge<0, 1, 2, 3> ---
        // 你的实现逻辑: _mm256_shuffle_ps(a.v[0], b.v[0], imm8)
        // 物理上 v[0] 包含了逻辑 Block 0 和 Block 1
        auto res = op::merge<0, 1, 2, 3>(batch_a, batch_b);
        op::store(result, res);

        // 2. 遍历所有逻辑块进行校验
        for (size_t c = 0; c < LCount; ++c) {
            size_t offset = c * LWidth;

            // 每一个逻辑块都应该独立且一致地执行了 shuffle 逻辑：
            // dst[0] = srcA[0], dst[1] = srcA[1], dst[2] = srcB[2], dst[3] = srcB[3]
            EXPECT_EQ(result[offset + 0], data_a[offset + 0]); // 逻辑索引 0
            EXPECT_EQ(result[offset + 1], data_a[offset + 1]); // 逻辑索引 1
            EXPECT_EQ(result[offset + 2], data_b[offset + 2]); // 逻辑索引 2
            EXPECT_EQ(result[offset + 3], data_b[offset + 3]); // 逻辑索引 3
        }

        // --- 3. 验证复杂的逻辑索引映射 merge<3, 0, 1, 2> ---
        // 预期: 逻辑槽位 01 来自 A 的 30, 逻辑槽位 23 来自 B 的 12
        auto res_shuffle = op::merge<3, 0, 1, 2>(batch_a, batch_b);
        op::store(result, res_shuffle);

        for (size_t c = 0; c < LCount; ++c) {
            size_t offset = c * LWidth;
            EXPECT_EQ(result[offset + 0], data_a[offset + 3]);
            EXPECT_EQ(result[offset + 1], data_a[offset + 0]);
            EXPECT_EQ(result[offset + 2], data_b[offset + 1]);
            EXPECT_EQ(result[offset + 3], data_b[offset + 2]);
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(merge_width4)
#endif

// ------------------------------------------ permute_logic_width4 ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void permute_logic_width4() noexcept
    {
        // 强制使用逻辑宽度为 4 的算子 (TYPE_T 可为 float32 或 float64)
        using op = KSIMD_DYN_PACKED_OP(TYPE_T, 4);
        
        constexpr size_t LCount = op::Count;      // 逻辑块数量 (e.g., AVX下float32为2, float64为1)
        constexpr size_t LWidth = op::Width;      // 逻辑宽度 (固定为 4)
        constexpr size_t TotalLanes = op::TotalLanes; 

        static_assert(LWidth == 4, "Permute test is designed for logical width 4");

        alignas(ALIGNMENT) TYPE_T data_in[TotalLanes];
        alignas(ALIGNMENT) TYPE_T result[TotalLanes];

        // 1. 初始化数据：每个逻辑块内部填充 [0, 1, 2, 3] 的偏移
        // 这样可以清晰地观察块内重排是否跨越了逻辑边界
        for (size_t c = 0; c < LCount; ++c) {
            for (size_t w = 0; w < LWidth; ++w) {
                data_in[c * LWidth + w] = TYPE_T(c * 10 + w); 
                // Block 0: 0, 1, 2, 3 | Block 1: 10, 11, 12, 13 ...
            }
        }

        auto batch_in = op::load(data_in);

        // --- 测试场景 1: 逆序重排 <3, 2, 1, 0> ---
        // 预期结果：每个 Block 内部变为 [3, 2, 1, 0]
        auto res_rev = op::permute<3, 2, 1, 0>(batch_in);
        op::store(result, res_rev);

        for (size_t c = 0; c < LCount; ++c) {
            size_t off = c * LWidth;
            EXPECT_EQ(result[off + 0], data_in[off + 3]);
            EXPECT_EQ(result[off + 1], data_in[off + 2]);
            EXPECT_EQ(result[off + 2], data_in[off + 1]);
            EXPECT_EQ(result[off + 3], data_in[off + 0]);
        }

        // --- 测试场景 2: 广播重排 <0, 0, 0, 0> ---
        // 预期结果：每个 Block 第一个元素广播到全块
        auto res_brd = op::permute<0, 0, 0, 0>(batch_in);
        op::store(result, res_brd);

        for (size_t c = 0; c < LCount; ++c) {
            size_t off = c * LWidth;
            EXPECT_EQ(result[off + 0], data_in[off + 0]);
            EXPECT_EQ(result[off + 1], data_in[off + 0]);
            EXPECT_EQ(result[off + 2], data_in[off + 0]);
            EXPECT_EQ(result[off + 3], data_in[off + 0]);
        }

        // --- 测试场景 3: 交叉选择 <1, 0, 3, 2> ---
        auto res_mix = op::permute<1, 0, 3, 2>(batch_in);
        op::store(result, res_mix);

        for (size_t c = 0; c < LCount; ++c) {
            size_t off = c * LWidth;
            EXPECT_EQ(result[off + 0], data_in[off + 1]);
            EXPECT_EQ(result[off + 1], data_in[off + 0]);
            EXPECT_EQ(result[off + 2], data_in[off + 3]);
            EXPECT_EQ(result[off + 3], data_in[off + 2]);
        }
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(permute_logic_width4)
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif

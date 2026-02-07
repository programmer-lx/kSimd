#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "fixed_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/fixed_op.hpp>

using namespace ksimd;

// ------------------------------------------ sequence ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sequence() noexcept
    {
        // 假设通过外部注入或上下文获取 TYPE_T 和具体的 WIDTH
        using op = KSIMD_DYN_FIXED_OP(TYPE_T, WIDTH);
        
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

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif

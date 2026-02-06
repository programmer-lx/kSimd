#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "fixed_op/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/fixed_op.hpp>

using namespace ksimd;

// ------------------------------------------ sequence ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sequence() noexcept
    {
        using op = KSIMD_DYN_FIXED_OP_COUNT(FLOAT_T, LANES, COUNT);

        constexpr size_t L = LANES;
        constexpr size_t C = COUNT;
        constexpr size_t Total = op::TotalLanes;

        alignas(ALIGNMENT) FLOAT_T out[Total];

        // 执行 sequence 生成
        auto v = op::sequence();
        op::store(out, v);

        // 验证逻辑：每一组内部应该是严格递增的 [0, 1, 2, 3]
        for (size_t i = 0; i < C; ++i)
        {
            for (size_t j = 0; j < L; ++j)
            {
                size_t global_idx = i * L + j;

                // 修正点：内存索引 j 处存放的值就是 j 本身
                FLOAT_T expected = static_cast<FLOAT_T>(j);

                EXPECT_EQ(out[global_idx], expected)
                    << "Sequence mismatch at Group " << i
                    << ", Local Lane " << j
                    << " (Global Index " << global_idx << ")";
            }
        }

        // 附加测试：验证带 base 的版本
        {
            constexpr FLOAT_T base = 10.0f;
            auto v_base = op::sequence(base);
            op::store(out, v_base);

            for (size_t i = 0; i < C; ++i) {
                for (size_t j = 0; j < L; ++j) {
                    size_t global_idx = i * L + j;
                    EXPECT_EQ(out[global_idx], static_cast<FLOAT_T>(j) + base);
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

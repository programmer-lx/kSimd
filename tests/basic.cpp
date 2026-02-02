#include "test.hpp"

#include <string>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "basic.cpp" // this file
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/base_op.hpp>

#pragma message("dispatch intrinsic: \"" KSIMD_STR("" KSIMD_DYN_FUNC_ATTR) "\"")

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR void kernel_dyn_impl(const float*, const size_t, float*) noexcept
    {
    }
}


#if KSIMD_ONCE

// export impl function
KSIMD_DYN_DISPATCH_FUNC(kernel_dyn_impl);

TEST(dyn_dispatch, pfn_table_size)
{
    EXPECT_EQ(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)), 5);
}

TEST(dyn_dispatch, float32)
{
    using namespace ksimd;

    // scalar
    {
        using trait = SimdTraits<SimdInstruction::Scalar, float32>;
        using batch_t = trait::batch_t;
        EXPECT_TRUE(alignof(batch_t) == alignof(float32));

        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::Scalar, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::Scalar);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == alignof(float32));
    }
    // sse
    {
        using trait = SimdTraits<SimdInstruction::SSE, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse2
    {
        using trait = SimdTraits<SimdInstruction::SSE2, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE2, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE2);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse3
    {
        using trait = SimdTraits<SimdInstruction::SSE3, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE3, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE3);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse4.1
    {
        using trait = SimdTraits<SimdInstruction::SSE4_1, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE4_1, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE4_1);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }

    // avx
    {
        using trait = SimdTraits<SimdInstruction::AVX, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 8);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
    // avx2
    {
        using trait = SimdTraits<SimdInstruction::AVX2, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX2, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX2);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 8);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
    // avx2+fma3
    {
        using trait = SimdTraits<SimdInstruction::AVX2_FMA3_F16C, float32>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX2_FMA3_F16C, float32>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX2_FMA3_F16C);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 4);
        EXPECT_TRUE(trait::Lanes == 8);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
}

TEST(dyn_dispatch, float64)
{
    using namespace ksimd;

    // scalar
    {
        using trait = SimdTraits<SimdInstruction::Scalar, float64>;
        using batch_t = trait::batch_t;
        EXPECT_TRUE(alignof(batch_t) == alignof(float64));

        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::Scalar, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::Scalar);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 2);
        EXPECT_TRUE(trait::BatchAlignment == alignof(float64));
    }
    // sse
    {
        using trait = SimdTraits<SimdInstruction::SSE, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 2);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse2
    {
        using trait = SimdTraits<SimdInstruction::SSE2, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE2, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE2);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 2);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse3
    {
        using trait = SimdTraits<SimdInstruction::SSE3, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE3, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE3);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 2);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }
    // sse4.1
    {
        using trait = SimdTraits<SimdInstruction::SSE4_1, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::SSE4_1, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::SSE4_1);
        EXPECT_TRUE(trait::BatchSize == 16);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 2);
        EXPECT_TRUE(trait::BatchAlignment == 16);
    }

    // avx
    {
        using trait = SimdTraits<SimdInstruction::AVX, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
    // avx2
    {
        using trait = SimdTraits<SimdInstruction::AVX2, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX2, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX2);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
    // avx2+fma3
    {
        using trait = SimdTraits<SimdInstruction::AVX2_FMA3_F16C, float64>;
        EXPECT_TRUE((std::is_same_v<BaseOp<SimdInstruction::AVX2_FMA3_F16C, float64>::traits, trait>));
        EXPECT_TRUE(trait::CurrentInstruction == SimdInstruction::AVX2_FMA3_F16C);
        EXPECT_TRUE(trait::BatchSize == 32);
        EXPECT_TRUE(trait::ElementSize == 8);
        EXPECT_TRUE(trait::Lanes == 4);
        EXPECT_TRUE(trait::BatchAlignment == 32);
    }
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
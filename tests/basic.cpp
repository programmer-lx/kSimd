#include "test.hpp"

#include <string>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "basic.cpp" // this file
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/op.hpp>

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
    EXPECT_EQ(std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_dyn_impl)), (size_t)ksimd::detail::SimdInstructionIndex::Num);
}

TEST(base_op, float32)
{
    using namespace ksimd;

    // scalar
    {
        using op = Op<SimdInstruction::Scalar, float32>;
        using batch_t = op::batch_t;
        static_assert(alignof(batch_t) == 16);

        static_assert((std::is_same_v<batch_t, vector_scalar::Batch<float32, 1>>));
        // static_assert(op::CurrentInstruction == SimdInstruction::Scalar);
        static_assert(op::BatchBytes == 16);
        static_assert(op::ElementBytes == 4);
        static_assert(op::TotalLanes == 4);
        static_assert(op::BatchAlignment == 16);
        static_assert(op::RegBytes == 16);
        static_assert(op::RegCount == 1);
        static_assert(op::RegLanes == 4);
    }
    // avx2+fma3
    {
        using op = Op<SimdInstruction::AVX2_FMA3_F16C, float32>;
        using batch_t = op::batch_t;
        static_assert(alignof(batch_t) == 32);
        static_assert((std::is_same_v<op::batch_t, x86_vector256::Batch<float32, 1>>));
        // static_assert(op::CurrentInstruction == SimdInstruction::AVX2_FMA3_F16C);
        static_assert(op::BatchBytes == 32);
        static_assert(op::ElementBytes == 4);
        static_assert(op::TotalLanes == 8);
        static_assert(op::BatchAlignment == 32);
        static_assert(op::RegBytes == 32);
        static_assert(op::RegCount == 1);
        static_assert(op::RegLanes == 8);
    }

    SUCCEED();
}

TEST(base_op, float64)
{
    using namespace ksimd;

    // scalar
    {
        using op = Op<SimdInstruction::Scalar, float64>;
        using batch_t = op::batch_t;
        static_assert(alignof(batch_t) == 16);

        static_assert((std::is_same_v<op::batch_t, vector_scalar::Batch<float64, 1>>));
        // static_assert(op::CurrentInstruction == SimdInstruction::Scalar);
        static_assert(op::BatchBytes == 16);
        static_assert(op::ElementBytes == 8);
        static_assert(op::TotalLanes == 2);
        static_assert(op::BatchAlignment == 16);
        static_assert(op::RegBytes == 16);
        static_assert(op::RegCount == 1);
        static_assert(op::RegLanes == 2);
    }
    // avx2+fma3
    {
        using op = Op<SimdInstruction::AVX2_FMA3_F16C, float64>;
        using batch_t = op::batch_t;
        static_assert(alignof(batch_t) == 32);
        static_assert((std::is_same_v<op::batch_t, x86_vector256::Batch<float64, 1>>));
        // static_assert(op::CurrentInstruction == SimdInstruction::AVX2_FMA3_F16C);
        static_assert(op::BatchBytes == 32);
        static_assert(op::ElementBytes == 8);
        static_assert(op::TotalLanes == 4);
        static_assert(op::BatchAlignment == 32);
        static_assert(op::RegBytes == 32);
        static_assert(op::RegCount == 1);
        static_assert(op::RegLanes == 4);
    }

    SUCCEED();
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
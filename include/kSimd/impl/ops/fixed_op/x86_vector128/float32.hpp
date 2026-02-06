#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

#define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV

KSIMD_NAMESPACE_BEGIN

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4, 1>
    : detail::Executor_SSE4_1_float32<1>
    , detail::Base_Mixin_SSE4_1_float32
    , FixedOpInfo<4, 1>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm_dp_ps(a.v[0], b.v[0], imm8) };
    }
};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, 4, 1>
    : FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4, 1>
{};

KSIMD_NAMESPACE_END

#undef KSIMD_API

#pragma once

#include "AVX2_float32.hpp"

KSIMD_NAMESPACE_BEGIN

// AVX2 + FMA指令特化
template<>
struct SimdOp<SimdInstruction::AVX2_FMA3_F16C, float32> : SimdOp<SimdInstruction::AVX2, float32>
{
    using traits = SimdTraits<SimdInstruction::AVX2_FMA3_F16C, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;

    KSIMD_OP_SIG_AVX2_FMA3_F16C(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm256_fmadd_ps(a.v, b.v, c.v) };
    }
};

KSIMD_NAMESPACE_END

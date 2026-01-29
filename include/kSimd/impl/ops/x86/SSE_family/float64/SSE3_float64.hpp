#pragma once

#include "SSE2_float64.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE3, float64> : SimdOp<SimdInstruction::SSE2, float64>
{
    using traits = SimdTraits<SimdInstruction::SSE3, float64>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;

    KSIMD_OP_SIG_SSE3(float64, reduce_sum, (batch_t v))
    {
        // input: [b, a]
        // hadd: [a+b]
        // get lane[0]
        __m128d result = _mm_hadd_pd(v.v, v.v);
        return _mm_cvtsd_f64(result);
    }
};

KSIMD_NAMESPACE_END

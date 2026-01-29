#pragma once

#include "SSE2_float32.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE3, float32> : SimdOp<SimdInstruction::SSE2, float32>
{
    using traits = SimdTraits<SimdInstruction::SSE3, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;

    KSIMD_OP_SIG_SSE3(float32, reduce_sum, (batch_t v))
    {
        // input: [d, c, b, a]
        // hadd: [c+d, a+b, c+d, a+b]
        // hadd: [a+b+c+d, .........]
        // get lane[0]

        __m128 result = _mm_hadd_ps(v.v, v.v);
        result = _mm_hadd_ps(result, result);
        return _mm_cvtss_f32(result);
    }
};

KSIMD_NAMESPACE_END

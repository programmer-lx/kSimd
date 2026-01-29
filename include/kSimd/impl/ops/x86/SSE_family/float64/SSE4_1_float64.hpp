#pragma once

#include "SSE3_float64.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE4_1, float64> : SimdOp<SimdInstruction::SSE3, float64>
{
    using traits = SimdTraits<SimdInstruction::SSE4_1, float64>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

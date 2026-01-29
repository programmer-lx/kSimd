#pragma once

#include "SSE3_float32.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE4_1, float32> : SimdOp<SimdInstruction::SSE3, float32>
{
    using traits = SimdTraits<SimdInstruction::SSE4_1, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

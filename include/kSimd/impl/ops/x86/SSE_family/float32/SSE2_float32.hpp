#pragma once

#include "SSE_float32.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE2, float32> : SimdOp<SimdInstruction::SSE, float32>
{
    using traits = SimdTraits<SimdInstruction::SSE2, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

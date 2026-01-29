#pragma once

#include "AVX_float64.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::AVX2, float64> : SimdOp<SimdInstruction::AVX, float64>
{
    using traits = SimdTraits<SimdInstruction::AVX2, float64>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

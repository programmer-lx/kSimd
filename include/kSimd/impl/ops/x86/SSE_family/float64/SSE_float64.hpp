#pragma once

#include "../_SSE_family_types.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE, float64> : detail::SimdOp_Scalar_FloatingPoint_Base<SimdInstruction::SSE, float64>
{
    using traits = SimdTraits<SimdInstruction::SSE, float64>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

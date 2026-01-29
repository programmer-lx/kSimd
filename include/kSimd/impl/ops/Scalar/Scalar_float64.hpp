#pragma once

#include "Scalar_base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::Scalar, float64> : detail::SimdOp_Scalar_FloatingPoint_Base<SimdInstruction::Scalar, float64>
{
};

KSIMD_NAMESPACE_END

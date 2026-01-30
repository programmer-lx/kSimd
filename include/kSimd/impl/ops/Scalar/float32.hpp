#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::Scalar, float32>
    : detail::SimdOp_Scalar_FloatingPoint_Base<SimdInstruction::Scalar, float32>
{
};

KSIMD_NAMESPACE_END

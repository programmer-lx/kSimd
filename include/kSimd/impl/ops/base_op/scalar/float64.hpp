#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct BaseOp<SimdInstruction::Scalar, float64>
    : detail::BaseOp_Scalar_FloatingPoint_Base<SimdInstruction::Scalar, float64>
{
};

KSIMD_NAMESPACE_END

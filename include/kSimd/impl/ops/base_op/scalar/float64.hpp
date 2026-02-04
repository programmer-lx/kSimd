#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64>
    : detail::BaseOp_Scalar_FloatingPoint_Base<
        SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
        vector_scalar::Batch<float64, 2, alignof(float64)>,
        vector_scalar::Mask<float64, 2, alignof(float64)>, alignof(float64)
    >
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64)
};

KSIMD_NAMESPACE_END

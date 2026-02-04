#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32>
    : detail::BaseOp_Scalar_FloatingPoint_Base<
        SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
        vector_scalar::Batch<float32, 4, alignof(float32)>,
        vector_scalar::Mask<float32, 4, alignof(float32)>, alignof(float32)
    >
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32)
};

KSIMD_NAMESPACE_END

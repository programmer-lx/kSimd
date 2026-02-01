#pragma once

#include "SimdTraits.inl"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction, is_scalar_type ScalarType>
struct SimdOp;

#define KSIMD_DYN_OP(scalar_type) \
    KSIMD_NAMESPACE_NAME::SimdOp<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, scalar_type>


KSIMD_NAMESPACE_END

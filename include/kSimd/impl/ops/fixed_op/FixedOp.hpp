#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction, is_scalar_type ScalarType, size_t Lanes>
struct FixedOp;

#define KSIMD_DYN_FIXED_OP(scalar_type, lanes) \
    KSIMD_NAMESPACE_NAME::FixedOp<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, scalar_type, lanes>


KSIMD_NAMESPACE_END
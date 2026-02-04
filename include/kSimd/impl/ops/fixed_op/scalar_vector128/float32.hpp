#pragma once

#include "kSimd/impl/ops/base_op/scalar/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32>
{};

KSIMD_NAMESPACE_END

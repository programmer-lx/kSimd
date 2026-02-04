#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct FixedOp<SimdInstruction::SSE, float32, 4>
{

};

KSIMD_NAMESPACE_END

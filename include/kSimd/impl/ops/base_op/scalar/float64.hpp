#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64>
    : BaseOpTraits_Scalar<float64, 1>
    , detail::Executor_Scalar_FloatingPoint_Base<BaseOpTraits_Scalar<float64, 1>>
    , detail::Base_Mixin_Scalar<float64, 1>
{};

KSIMD_NAMESPACE_END

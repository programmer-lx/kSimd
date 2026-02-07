#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32>
    : BaseOpTraits_Scalar<float32, 1>
    , detail::Executor_Scalar_float32<BaseOpTraits_Scalar<float32, 1>>
    , detail::Base_Mixin_Scalar<float32, 1>
{};

KSIMD_NAMESPACE_END

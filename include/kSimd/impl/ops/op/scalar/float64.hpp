#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_TRAITS BaseOpTraits_Scalar<float64, 1>
template<>
struct Op<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64>
    : KSIMD_TRAITS

    // executor
    , detail::Executor_Scalar_float64<KSIMD_TRAITS>

    // horizontal mixin
    , detail::Base_Mixin_Scalar<KSIMD_TRAITS>
{};
#undef KSIMD_TRAITS

KSIMD_NAMESPACE_END

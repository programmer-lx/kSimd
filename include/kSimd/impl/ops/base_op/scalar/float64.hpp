#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    template<size_t reg_count>
    struct Executor_Scalar_float64
        : detail::Executor_Scalar_FloatingPoint_Base<
            SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
            vector_scalar::Batch<float64, reg_count, alignof(float64)>,
            vector_scalar::Mask<float64, reg_count, alignof(float64)>, alignof(float64)
        >
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_Scalar<float64, reg_count>)
    };
}

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64>
    : detail::Executor_Scalar_float64<1>
    , detail::Base_Mixin_Scalar<float64, 1, alignof(float64)>
{};

KSIMD_NAMESPACE_END

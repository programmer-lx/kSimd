#pragma once

#include "base.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    template<size_t reg_count>
    struct Executor_Scalar_float64
        : detail::Executor_Scalar_FloatingPoint_Base<
            SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
            vector_scalar::Batch<float64, reg_count>,
            vector_scalar::Mask<float64, reg_count>, alignment::Vec128
        >
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_Scalar<float64, reg_count>)
    };
}

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float64>
    : detail::Executor_Scalar_float64<1>
    , detail::Base_Mixin_Scalar_sequence<float64, 1>
    , detail::Base_Mixin_Scalar_reduce_add<float64, 1>
{};

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/scalar.hpp"

KSIMD_NAMESPACE_BEGIN

template<is_scalar_type S>
struct BaseOpTraits_Scalar
    : detail::SimdTraits_Base<
        SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
        vector_scalar::Batch<S, 16 / sizeof(S), alignof(S)>,    // vector128
        vector_scalar::Mask<S, 16 / sizeof(S), alignof(S)>,     // vector128
        alignof(S)
    >
{};

KSIMD_NAMESPACE_END

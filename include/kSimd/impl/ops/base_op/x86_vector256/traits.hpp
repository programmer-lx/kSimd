#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"

KSIMD_NAMESPACE_BEGIN

template<is_scalar_type S>
struct BaseOpTraits_AVX_Family
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, x86_vector256::Batch<S, 1>, x86_vector256::Mask<S, 1>, alignment::Vec256>
{};

KSIMD_NAMESPACE_END

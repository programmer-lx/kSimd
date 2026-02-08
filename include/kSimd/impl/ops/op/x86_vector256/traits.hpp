#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"

KSIMD_NAMESPACE_BEGIN

template<is_scalar_type S, size_t RegCount, typename MaskType>
struct BaseOpTraits_AVX_Family
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C, x86_vector256::Batch<S, RegCount>,
                              MaskType, alignment::Vec256>
{};

KSIMD_NAMESPACE_END

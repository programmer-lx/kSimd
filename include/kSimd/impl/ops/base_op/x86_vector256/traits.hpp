#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I, is_scalar_type S, size_t RegCount>
struct BaseOpTraits_AVX_Family
    : detail::SimdTraits_Base<I, x86_vector256::Batch<S, RegCount>,
                              x86_vector256::Mask<S, RegCount>, alignment::Vec256>
{};

KSIMD_NAMESPACE_END

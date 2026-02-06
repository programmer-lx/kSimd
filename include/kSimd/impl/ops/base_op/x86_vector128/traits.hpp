#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"

KSIMD_NAMESPACE_BEGIN

// SSE2+
template<SimdInstruction I, is_scalar_type S, size_t RegCount>
struct BaseOpTraits_SSE2_Plus
    : detail::SimdTraits_Base<I, x86_vector128::Batch<S, RegCount>, x86_vector128::Mask<S, RegCount>, alignment::Vec128>
{};

KSIMD_NAMESPACE_END

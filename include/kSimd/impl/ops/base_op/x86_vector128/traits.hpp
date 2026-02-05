#pragma once

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
    #include "kSimd/impl/ops/base_op/scalar/traits.hpp"
#endif

KSIMD_NAMESPACE_BEGIN

template<is_scalar_type S>
struct BaseOpTraits_SSE;

// SSE
template<is_scalar_type S>
    requires std::is_same_v<float32, S> // float32 only
struct BaseOpTraits_SSE<S>
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, x86_vector128::Batch<S, 1>,
                              x86_vector128::Mask<S, 1>, alignment::Vec128>
{};


// SSE 其他类型 使用标量模拟
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
template<is_scalar_type S>
    requires(!std::is_same_v<float32, S>) // NOT float32
struct BaseOpTraits_SSE<S>
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE,
                              vector_scalar::Batch<S, 16 / sizeof(S), alignment::Vec128>,
                              vector_scalar::Mask<S, 16 / sizeof(S), alignment::Vec128>, alignment::Vec128>
{};
#endif

// SSE2+
template<is_scalar_type S>
struct BaseOpTraits_SSE2_Plus
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, x86_vector128::Batch<S, 1>, x86_vector128::Mask<S, 1>, alignment::Vec128>
{};

KSIMD_NAMESPACE_END

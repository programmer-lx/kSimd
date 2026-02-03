#pragma once

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "kSimd/impl/ops/base_op/scalar/base.hpp"
#endif

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector128
{
    template<is_scalar_type scalar_type>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128i;
        static constexpr size_t byte_size = 16;

        __m128i v; // (u)int(n) 128bits
    };

    template<>
    struct Batch<float32>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128;
        static constexpr size_t byte_size = 16;

        __m128 v;
    };

    template<>
    struct Batch<float64>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128d;
        static constexpr size_t byte_size = 16;

        __m128d v;
    };

    template<is_scalar_type scalar_type>
    struct Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128i;

        __m128i m;
    };

    template<>
    struct Mask<float32>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128;

        __m128 m;
    };

    template<>
    struct Mask<float64>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128d;

        __m128d m;
    };
}

// traits
// SSE
template<is_scalar_type S>
    requires std::is_same_v<float32, S> // float32 only
struct BaseOpTraits<SimdInstruction::SSE, S>
    : detail::SimdTraits_Base<SimdInstruction::SSE, S, x86_vector128::Batch<S>, x86_vector128::Mask<S>, alignment::Vec128>
{
};


#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
template<is_scalar_type S>
    requires (!std::is_same_v<float32, S>) // NOT float32
struct BaseOpTraits<SimdInstruction::SSE, S>
    : detail::SimdTraits_Base<
        SimdInstruction::SSE, S, vector_scalar::Batch<S, alignment::Vec128>, vector_scalar::Mask<S, alignment::Vec128>, alignment::Vec128>
{
};
#endif

// SSE2+
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && std::is_same_v<float32, S>) // float32 only
struct BaseOpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, S, x86_vector128::Batch<S>, x86_vector128::Mask<S>, alignment::Vec128>
{
};

template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && !std::is_same_v<float32, S>) // NOT float32
struct BaseOpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, S, x86_vector128::Batch<S>, x86_vector128::Mask<S>, alignment::Vec128>
{
};

KSIMD_NAMESPACE_END
#pragma once

#include <immintrin.h> // AVX

#include "../../dispatch.hpp"

KSIMD_NAMESPACE_BEGIN

namespace AVX_family
{
    template<is_scalar_type scalar_type>
    struct Batch
    {
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256i;

        __m256i v;
    };

    template<>
    struct Batch<float32>
    {
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256;

        __m256 v;
    };

    template<>
    struct Batch<float64>
    {
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256d;

        __m256d v;
    };
}

// traits
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction > SimdInstruction::AVX_Start && Instruction < SimdInstruction::AVX_End)
struct SimdTraits<Instruction, S> : detail::SimdTraits_Base<Instruction, S, AVX_family::Batch<S>, Alignment::AVX_Family>
{
};

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector256
{
    template<is_scalar_type scalar_type>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256i;
        static constexpr size_t byte_size = 32;

        __m256i v;
    };

    template<>
    struct Batch<float32>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256;
        static constexpr size_t byte_size = 32;

        __m256 v;
    };

    template<>
    struct Batch<float64>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256d;
        static constexpr size_t byte_size = 32;

        __m256d v;
    };

    template<is_scalar_type scalar_type>
    struct Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256i;

        __m256i m;
    };

    template<>
    struct Mask<float32>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256;

        __m256 m;
    };

    template<>
    struct Mask<float64>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256d;

        __m256d m;
    };
}

// traits
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction > SimdInstruction::AVX_Start && Instruction < SimdInstruction::AVX_End)
struct OpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, x86_vector256::Batch<S>, x86_vector256::Mask<S>, alignment::Vec256>
{
};

KSIMD_NAMESPACE_END

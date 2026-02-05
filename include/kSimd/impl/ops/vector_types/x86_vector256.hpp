#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector256
{
    template<is_scalar_type scalar_type, size_t reg_count>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256i;
        static constexpr size_t byte_size = 32 * reg_count;

        __m256i v[reg_count];
    };

    template<size_t reg_count>
    struct Batch<float32, reg_count>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256;
        static constexpr size_t byte_size = 32 * reg_count;

        __m256 v[reg_count];
    };

    template<size_t reg_count>
    struct Batch<float64, reg_count>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256d;
        static constexpr size_t byte_size = 32 * reg_count;

        __m256d v[reg_count];
    };

    template<is_scalar_type scalar_type, size_t reg_count>
    struct Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256i;

        __m256i m[reg_count];
    };

    template<size_t reg_count>
    struct Mask<float32, reg_count>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256;

        __m256 m[reg_count];
    };

    template<size_t reg_count>
    struct Mask<float64, reg_count>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256d;

        __m256d m[reg_count];
    };
}

// traits
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction > SimdInstruction::AVX_Start && Instruction < SimdInstruction::AVX_End)
struct OpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, x86_vector256::Batch<S, 1>, x86_vector256::Mask<S, 1>, alignment::Vec256>
{
};

KSIMD_NAMESPACE_END

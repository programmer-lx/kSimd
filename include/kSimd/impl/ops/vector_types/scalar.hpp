#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace vector_scalar
{
    template<is_scalar_type S, size_t Lanes, size_t Alignment>
    struct Batch
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = sizeof(S) * Lanes;

        alignas(Alignment) S v[Lanes];

        static_assert(sizeof(v) == sizeof(S) * Lanes);
    };

    template<is_scalar_type S, size_t Lanes, size_t alignment>
    struct Mask
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::ScalarArray;

        alignas(alignment) S m[Lanes];

        static_assert(sizeof(m) == sizeof(S) * Lanes);
    };
}

template<is_scalar_type S>
struct OpTraits<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, S>
    : detail::SimdTraits_Base<
        SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR,
        vector_scalar::Batch<S, 16 / sizeof(S), alignof(S)>,    // vector128
        vector_scalar::Mask<S, 16 / sizeof(S), alignof(S)>,     // vector128
        alignof(S)
    >
{};

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace vector_scalar
{
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t VectorByteSize = 16;

    template<is_scalar_type S, size_t Alignment>
    struct Batch
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = VectorByteSize;

        alignas(Alignment) S v[VectorByteSize / sizeof(S)];

        static_assert(sizeof(v) == VectorByteSize);
    };

    template<is_scalar_type S, size_t alignment>
    struct Mask
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::ScalarArray;

        alignas(alignment) S m[VectorByteSize / sizeof(S)];

        static_assert(sizeof(m) == VectorByteSize);
    };
}

template<is_scalar_type S>
struct BaseOpTraits<SimdInstruction::Scalar, S>
    : detail::SimdTraits_Base<
        SimdInstruction::Scalar,
        S,
        vector_scalar::Batch<S, alignof(S)>,    // vector128
        vector_scalar::Mask<S, alignof(S)>,     // vector128
        alignof(S)
    >
{
};

KSIMD_NAMESPACE_END

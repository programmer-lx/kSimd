#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace base_vector_scalar
{
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t FullByteSize = 16;

    template<is_scalar_type S, size_t alignment>
    struct Batch
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = FullByteSize;

        alignas(alignment) S v[FullByteSize / sizeof(S)];

        static_assert(sizeof(v) == FullByteSize, "Scalar type is 128bits vector");
    };

    template<is_scalar_type S, size_t alignment>
    struct Mask
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::ScalarArray;

        alignas(alignment) S m[FullByteSize / sizeof(S)];

        static_assert(sizeof(m) == FullByteSize, "Scalar type is 128bits vector");
    };
}

template<is_scalar_type S>
struct BaseOpTraits<SimdInstruction::Scalar, S>
    : detail::SimdTraits_Base<
        SimdInstruction::Scalar,
        S,
        base_vector_scalar::Batch<S, alignof(S)>,
        base_vector_scalar::Mask<S, alignof(S)>,
        alignof(S)
    >
{
};

KSIMD_NAMESPACE_END

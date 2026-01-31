#pragma once

#include "../dispatch.hpp"

KSIMD_NAMESPACE_BEGIN

namespace Scalar_family
{
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t ByteSize = 16;

    template<is_scalar_type scalar_type, size_t alignment>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = ByteSize;

        alignas(alignment) scalar_type v[ByteSize / sizeof(scalar_type)];

        static_assert(sizeof(v) == ByteSize, "Scalar type is 128bits vector");
    };
}

template<is_scalar_type S>
struct SimdTraits<SimdInstruction::Scalar, S> : detail::SimdTraits_Base<SimdInstruction::Scalar, S, Scalar_family::Batch<S, alignof(S)>, alignof(S)>
{
};

KSIMD_NAMESPACE_END

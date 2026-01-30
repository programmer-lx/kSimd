#pragma once

#include "../dispatch.hpp"

KSIMD_NAMESPACE_BEGIN

namespace Scalar_family
{
    template<is_scalar_type scalar_type, size_t alignment>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = 16;

        alignas(alignment) scalar_type v[16 / sizeof(scalar_type)];

        static_assert(sizeof(v) == 16, "Scalar type is 128bits vector");
    };
}

template<is_scalar_type S>
struct SimdTraits<SimdInstruction::Scalar, S> : detail::SimdTraits_Base<SimdInstruction::Scalar, S, Scalar_family::Batch<S, alignof(S)>, alignof(S)>
{
};

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace vector_scalar
{
    template<is_scalar_type S, size_t RegCount, size_t Alignment>
    struct Batch
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = sizeof(S) * RegCount;
        static constexpr size_t reg_count = RegCount;

        alignas(Alignment) S v[RegCount];

        static_assert(sizeof(v) == sizeof(S) * RegCount);
    };

    template<is_scalar_type S, size_t RegCount, size_t alignment>
    struct Mask
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::ScalarArray;
        static constexpr size_t reg_count = RegCount;

        alignas(alignment) S m[RegCount];

        static_assert(sizeof(m) == sizeof(S) * RegCount);
    };
}

KSIMD_NAMESPACE_END

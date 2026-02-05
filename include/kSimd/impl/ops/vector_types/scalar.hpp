#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace vector_scalar
{
    // 标量特殊处理，视 128bit 为一个寄存器的容量，向SSE看齐
    template<is_scalar_type S, size_t RegCount, size_t Alignment>
    struct Batch
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::ScalarArray;
        static constexpr size_t byte_size = 16 * RegCount;
        static constexpr size_t reg_count = RegCount;

        alignas(Alignment) S v[(16 / sizeof(S)) * RegCount]; // vector128 * RegCount

        static_assert(sizeof(v) == 16 * RegCount);
    };

    template<is_scalar_type S, size_t RegCount, size_t Alignment>
    struct Mask
    {
        using scalar_t = S;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::ScalarArray;
        static constexpr size_t reg_count = RegCount;

        alignas(Alignment) S m[(16 / sizeof(S)) * RegCount];

        static_assert(sizeof(m) == 16 * RegCount);
    };
}

KSIMD_NAMESPACE_END

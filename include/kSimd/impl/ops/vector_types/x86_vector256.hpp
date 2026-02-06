#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector256
{
    template<is_scalar_type scalar_type, size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256i;
        static constexpr size_t byte_size = 32 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m256i v[RegCount];
    };

    template<size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Batch<float32, RegCount>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256;
        static constexpr size_t byte_size = 32 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m256 v[RegCount];
    };

    template<size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Batch<float64, RegCount>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m256d;
        static constexpr size_t byte_size = 32 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m256d v[RegCount];
    };

    template<is_scalar_type scalar_type, size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256i;
        static constexpr size_t reg_count = RegCount;

        __m256i m[RegCount];
    };

    template<size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Mask<float32, RegCount>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256;
        static constexpr size_t reg_count = RegCount;

        __m256 m[RegCount];
    };

    template<size_t RegCount>
    struct alignas(alignment::clamp(alignment::Vec256 * RegCount)) Mask<float64, RegCount>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m256d;
        static constexpr size_t reg_count = RegCount;

        __m256d m[RegCount];
    };
}

KSIMD_NAMESPACE_END

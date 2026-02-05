#pragma once

// SSE 除了float32的类型 使用标量模拟
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
    #include "kSimd/impl/ops/base_op/scalar/base.hpp"
#endif

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector128
{
    template<is_scalar_type scalar_type, size_t RegCount>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128i;
        static constexpr size_t byte_size = 16 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m128i v[RegCount]; // (u)int(n)
    };

    template<size_t RegCount>
    struct Batch<float32, RegCount>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128;
        static constexpr size_t byte_size = 16 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m128 v[RegCount];
    };

    template<size_t RegCount>
    struct Batch<float64, RegCount>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128d;
        static constexpr size_t byte_size = 16 * RegCount;
        static constexpr size_t reg_count = RegCount;

        __m128d v[RegCount];
    };

    template<is_scalar_type scalar_type, size_t RegCount>
    struct Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128i;
        static constexpr size_t reg_count = RegCount;

        __m128i m[RegCount];
    };

    template<size_t RegCount>
    struct Mask<float32, RegCount>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128;
        static constexpr size_t reg_count = RegCount;

        __m128 m[RegCount];
    };

    template<size_t RegCount>
    struct Mask<float64, RegCount>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128d;
        static constexpr size_t reg_count = RegCount;

        __m128d m[RegCount];
    };
} // namespace x86_vector128

KSIMD_NAMESPACE_END

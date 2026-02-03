#pragma once

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "kSimd/impl/ops/Scalar/base.hpp"
#endif

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace vector128_x86
{
    namespace SSE
    {
        // 除了float32，其他类型直接fallback到标量
        template<is_scalar_type scalar_type>
        struct Batch;

        template<>
        struct Batch<float32>
        {
            using scalar_t = float32;
            static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128;
            static constexpr size_t byte_size = 16;

            __m128 v;
        };

        // 除了float32，其他类型fallback到标量
        template<is_scalar_type scalar_type>
        struct Mask;

        template<>
        struct Mask<float32>
        {
            using scalar_t = float32;
            static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128;

            __m128 m;
        };
    }

    namespace SSE2_up
    {
        template<is_scalar_type scalar_type>
        struct Batch
        {
            static_assert(!std::is_same_v<scalar_type, float32>, "Batch<float32> is SSE type, not SSE2_up");

            using scalar_t = scalar_type;
            static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128i;
            static constexpr size_t byte_size = 16;

            __m128i v; // (u)int(n) 128bits
        };

        template<>
        struct Batch<float64>
        {
            using scalar_t = float64;
            static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128d;
            static constexpr size_t byte_size = 16;

            __m128d v;
        };

        template<is_scalar_type scalar_type>
        struct Mask
        {
            static_assert(!std::is_same_v<scalar_type, float32>, "Mask<float32> is SSE type, not SSE2_up");

            using scalar_t = scalar_type;
            static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128i;

            __m128i m;
        };

        template<>
        struct Mask<float64>
        {
            using scalar_t = float64;
            static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128d;

            __m128d m;
        };
    }
}

// traits
// SSE
template<is_scalar_type S>
    requires std::is_same_v<float32, S> // float32 only
struct BaseOpTraits<SimdInstruction::SSE, S>
    : detail::SimdTraits_Base<SimdInstruction::SSE, S, vector128_x86::SSE::Batch<S>, vector128_x86::SSE::Mask<S>, alignment::SSE_Family>
{
};


#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
template<is_scalar_type S>
    requires (!std::is_same_v<float32, S>) // NOT float32
struct BaseOpTraits<SimdInstruction::SSE, S>
    : detail::SimdTraits_Base<
        SimdInstruction::SSE, S, vector_scalar::Batch<S, alignment::SSE_Family>, vector_scalar::Mask<S, alignment::SSE_Family>, alignment::SSE_Family>
{
};
#endif

// SSE2+
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && std::is_same_v<float32, S>) // float32 only
struct BaseOpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, S, vector128_x86::SSE::Batch<S>, vector128_x86::SSE::Mask<S>, alignment::SSE_Family>
{
};

template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && !std::is_same_v<float32, S>) // NOT float32
struct BaseOpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, S, vector128_x86::SSE2_up::Batch<S>, vector128_x86::SSE2_up::Mask<S>, alignment::SSE_Family>
{
};

KSIMD_NAMESPACE_END
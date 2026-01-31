#pragma once

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <smmintrin.h> // SSE4.1

#include "../../Scalar/base.hpp"

#include "../../dispatch.hpp"

KSIMD_NAMESPACE_BEGIN

namespace SSE_family
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
    }
}

// traits
// SSE
template<is_scalar_type S>
    requires std::is_same_v<float32, S> // float32 only
struct SimdTraits<SimdInstruction::SSE, S> : detail::SimdTraits_Base<SimdInstruction::SSE, S, SSE_family::SSE::Batch<S>, alignment::SSE_Family>
{
};

template<is_scalar_type S>
    requires (!std::is_same_v<float32, S>) // NOT float32
struct SimdTraits<SimdInstruction::SSE, S> : detail::SimdTraits_Base<SimdInstruction::SSE, S, Scalar_family::Batch<S, alignment::SSE_Family>, alignment::SSE_Family>
{
};

// SSE2+
template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && std::is_same_v<float32, S>) // float32 only
struct SimdTraits<Instruction, S> : detail::SimdTraits_Base<Instruction, S, SSE_family::SSE::Batch<S>, alignment::SSE_Family>
{
};

template<SimdInstruction Instruction, is_scalar_type S>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End && !std::is_same_v<float32, S>) // NOT float32
struct SimdTraits<Instruction, S> : detail::SimdTraits_Base<Instruction, S, SSE_family::SSE2_up::Batch<S>, alignment::SSE_Family>
{
};

KSIMD_NAMESPACE_END
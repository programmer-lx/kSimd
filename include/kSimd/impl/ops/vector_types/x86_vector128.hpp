#pragma once

// SSE 除了float32的类型 使用标量模拟
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
    #include "kSimd/impl/ops/base_op/scalar/base.hpp"
#endif

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace x86_vector128
{
    template<is_scalar_type scalar_type, size_t reg_count>
    struct Batch
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128i;
        static constexpr size_t byte_size = 16 * reg_count;

        __m128i v[reg_count]; // (u)int(n)
    };

    template<size_t reg_count>
    struct Batch<float32, reg_count>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128;
        static constexpr size_t byte_size = 16 * reg_count;

        __m128 v[reg_count];
    };

    template<size_t reg_count>
    struct Batch<float64, reg_count>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingSimdType underlying_simd_type = detail::UnderlyingSimdType::m128d;
        static constexpr size_t byte_size = 16 * reg_count;

        __m128d v[reg_count];
    };

    template<is_scalar_type scalar_type, size_t reg_count>
    struct Mask
    {
        using scalar_t = scalar_type;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128i;

        __m128i m[reg_count];
    };

    template<size_t reg_count>
    struct Mask<float32, reg_count>
    {
        using scalar_t = float32;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128;

        __m128 m[reg_count];
    };

    template<size_t reg_count>
    struct Mask<float64, reg_count>
    {
        using scalar_t = float64;
        static constexpr detail::UnderlyingMaskType underlying_mask_type = detail::UnderlyingMaskType::m128d;

        __m128d m[reg_count];
    };
} // namespace x86_vector128

// SSE
template<is_scalar_type S>
    requires std::is_same_v<float32, S> // float32 only
struct OpTraits<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, S>
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, x86_vector128::Batch<S, 1>,
                              x86_vector128::Mask<S, 1>, alignment::Vec128>
{};


// SSE 其他类型 使用标量模拟
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
template<is_scalar_type S>
    requires(!std::is_same_v<float32, S>) // NOT float32
struct OpTraits<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, S>
    : detail::SimdTraits_Base<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE,
                              vector_scalar::Batch<S, 16 / sizeof(S), alignment::Vec128>,
                              vector_scalar::Mask<S, 16 / sizeof(S), alignment::Vec128>, alignment::Vec128>
{};
#endif

// SSE2+
template<SimdInstruction Instruction, is_scalar_type S>
    requires(Instruction >= SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2 && Instruction < SimdInstruction::SSE_End)
struct OpTraits<Instruction, S>
    : detail::SimdTraits_Base<Instruction, x86_vector128::Batch<S, 1>, x86_vector128::Mask<S, 1>, alignment::Vec128>
{};

KSIMD_NAMESPACE_END

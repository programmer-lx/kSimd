#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/base_op/x86_vector256/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_BATCH_T x86_vector128::Batch<float32, 2>
namespace detail
{
    #define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
    struct Fixed_Mixin_SSE4_1_sequence_float32_4x2
    {
        KSIMD_API(KSIMD_BATCH_T) sequence() noexcept
        {
            return { _mm_set_ps(3, 2, 1, 0), _mm_set_ps(3, 2, 1, 0) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base) noexcept
        {
            __m128 base_v = _mm_set1_ps(base);
            __m128 seq1 = _mm_set_ps(3, 2, 1, 0);
            __m128 seq2 = _mm_set_ps(3, 2, 1, 0);
            return { _mm_add_ps(seq1, base_v), _mm_add_ps(seq2, base_v) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base, float32 stride) noexcept
        {
            __m128 base_v = _mm_set1_ps(base);
            __m128 stride_v = _mm_set1_ps(stride);
            __m128 seq1 = _mm_set_ps(3, 2, 1, 0);
            __m128 seq2 = _mm_set_ps(3, 2, 1, 0);
            __m128 t = _mm_mul_ps(stride_v, base_v);
            return { _mm_add_ps(t, seq1), _mm_add_ps(t, seq2) };
        }
    };
    struct Fixed_Mixin_SSSE4_1_dot_float32_4x2
    {
        template<int src_mask, int dst_mask>
        KSIMD_API(KSIMD_BATCH_T) dot(KSIMD_BATCH_T a, KSIMD_BATCH_T b) noexcept
        {
            constexpr int imm8 = (src_mask << 4) | dst_mask;
            return { _mm_dp_ps(a.v[0], b.v[0], imm8), _mm_dp_ps(a.v[1], b.v[1], imm8) };
        }
    };
    #undef KSIMD_API

#undef KSIMD_BATCH_T

// AVX+ mixin horizontal functions
#define KSIMD_BATCH_T x86_vector256::Batch<float32, 1>

    #define KSIMD_API(...) KSIMD_OP_AVX2_FMA3_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Fixed_Mixin_AVX2_FMA3_dot_float32_4x2
    {
        template<int src_mask, int dst_mask>
        KSIMD_API(KSIMD_BATCH_T) dot(
            KSIMD_BATCH_T a,
            KSIMD_BATCH_T b
        ) noexcept
        {
            constexpr int imm8 = (src_mask << 4) | dst_mask;
            return { _mm256_dp_ps(a.v[0], b.v[0], imm8) };
        }
    };
    struct Fixed_Mixin_AVX2_FMA3_sequence_float32_4x2
    {
        KSIMD_API(KSIMD_BATCH_T) sequence() noexcept
        {
            return { _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base) noexcept
        {
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0);
            return { _mm256_add_ps(iota, base_v) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base, float32 stride) noexcept
        {
            __m256 stride_v = _mm256_set1_ps(stride);
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0);
            return { _mm256_fmadd_ps(stride_v, iota, base_v) };
        }
    };
    #undef KSIMD_API
}
#undef KSIMD_BATCH_T


template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4, 2>
    : detail::Executor_SSE4_1_float32<2>
    , detail::Fixed_Mixin_SSE4_1_sequence_float32_4x2
    , detail::Fixed_Mixin_SSSE4_1_dot_float32_4x2
    , FixedOpInfo<4, 2>
    , FixedOpHelper<4>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, 4, 2>
    : detail::Executor_AVX2_FMA3_float32<1>
    , detail::Fixed_Mixin_AVX2_FMA3_sequence_float32_4x2
    , detail::Fixed_Mixin_AVX2_FMA3_dot_float32_4x2
    , FixedOpHelper<4>
    , FixedOpInfo<4, 2>
{};

KSIMD_NAMESPACE_END

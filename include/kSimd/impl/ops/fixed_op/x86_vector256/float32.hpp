#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/base_op/x86_vector256/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4, 2>
    : detail::Executor_SSE4_1_float32<2>
    , FixedOpInfo<4, 2>
    , FixedOpHelper<4>
{
    KSIMD_API(batch_t) sequence() noexcept
    {
        __m128 iota = _mm_set_ps(3, 2, 1, 0);
        return { iota, iota };
    }

    KSIMD_API(batch_t) sequence(float32 base) noexcept
    {
        __m128 iota = _mm_set_ps(3, 2, 1, 0);
        __m128 base_v = _mm_set1_ps(base);
        __m128 res = _mm_add_ps(iota, base_v);
        return { res, res };
    }

    KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
    {
        __m128 iota = _mm_set_ps(3, 2, 1, 0);
        __m128 base_v = _mm_set1_ps(base);
        __m128 stride_v = _mm_set1_ps(stride);
        __m128 res = _mm_add_ps(base_v, _mm_mul_ps(stride_v, iota));
        return { res, res };
    }

    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm_dp_ps(a.v[0], b.v[0], imm8), _mm_dp_ps(a.v[1], b.v[1], imm8) };
    }
};
#undef KSIMD_API

#define KSIMD_API(...) KSIMD_OP_AVX2_FMA3_API static __VA_ARGS__ KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, 4, 2>
    : detail::Executor_AVX2_FMA3_float32<1>
    , FixedOpHelper<4>
    , FixedOpInfo<4, 2>
{
    KSIMD_API(batch_t) sequence() noexcept
    {
        return { _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0) };
    }

    KSIMD_API(batch_t) sequence(float32 base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0);
        return { _mm256_add_ps(iota, base_v) };
    }

    KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(3, 2, 1, 0, 3, 2, 1, 0);
        return { _mm256_fmadd_ps(stride_v, iota, base_v) };
    }

    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm256_dp_ps(a.v[0], b.v[0], imm8) };
    }
};
#undef KSIMD_API

KSIMD_NAMESPACE_END

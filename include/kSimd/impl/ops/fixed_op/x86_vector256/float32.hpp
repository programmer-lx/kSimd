#pragma once

#include "kSimd/impl/ops/base_op/x86_vector256/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_API(...) KSIMD_OP_AVX2_FMA3_F16C_API static __VA_ARGS__ KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C, float32, 4, 2>
    : BaseOpTraits_AVX_Family<float32, 1>
    , detail::Executor_AVX2_FMA3_F16C_float32<BaseOpTraits_AVX_Family<float32, 1>, 1>
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

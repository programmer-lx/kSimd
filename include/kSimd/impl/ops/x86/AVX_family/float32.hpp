#pragma once

#include "_AVX_family_types.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I>
    requires (I >= SimdInstruction::AVX && I <= SimdInstruction::AVX2)
struct SimdOp<I, float32>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::AVX, float32)

    KSIMD_OP_SIG_AVX(batch_t, load, (const float32* mem))
    {
        return { _mm256_load_ps(mem) };
    }

    KSIMD_OP_SIG_AVX(batch_t, loadu, (const float32* mem))
    {
        return { _mm256_loadu_ps(mem) };
    }

    KSIMD_OP_SIG_AVX(void, store, (float32* mem, batch_t v))
    {
        _mm256_store_ps(mem, v.v);
    }

    KSIMD_OP_SIG_AVX(void, storeu, (float32* mem, batch_t v))
    {
        _mm256_storeu_ps(mem, v.v);
    }

    KSIMD_OP_SIG_AVX(batch_t, zero, ())
    {
        return { _mm256_setzero_ps() };
    }

    KSIMD_OP_SIG_AVX(batch_t, set, (float32 x))
    {
        return { _mm256_set1_ps(x) };
    }

    KSIMD_OP_SIG_AVX(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, one_div, (batch_t v))
    {
        return { _mm256_rcp_ps(v.v) };
    }

    KSIMD_OP_SIG_AVX(float32, reduce_sum, (batch_t v))
    {
        // [8, 7, 6, 5, 4, 3, 2, 1]
        // hadd
        // [78, 56, 78, 56, 34, 12, 34, 12]
        // hadd
        // [5678, 5678, 5678, 5678, 1234, 1234, 1234, 1234]
        __m256 t1 = _mm256_hadd_ps(v.v, v.v);
        t1 = _mm256_hadd_ps(t1, t1);

        // low = [1234, 1234, 1234, 1234]
        // high = [5678, 5678, 5678, 5678]
        __m128 low = _mm256_castps256_ps128(t1);
        __m128 high = _mm256_extractf128_ps(t1, 0b1);

        // add
        // [12345678, ....]
        // get lane[0]
        low = _mm_add_ps(low, high);
        return _mm_cvtss_f32(low);
    }

    KSIMD_OP_SIG_AVX(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm256_add_ps(_mm256_mul_ps(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sqrt, (batch_t v))
    {
        return { _mm256_sqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, rsqrt, (batch_t v))
    {
        return { _mm256_rsqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, abs, (batch_t v))
    {
        // 将 sign bit 反转即可
        return { _mm256_and_ps(v.v, _mm256_set1_ps(sign_bit_clear_mask<float>)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_min_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_max_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_ORD_Q) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_not, (batch_t v))
    {
        return { _mm256_xor_ps(v.v, _mm256_set1_ps(one_block<float>)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_andnot_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm256_or_ps(_mm256_and_ps(mask.v, a.v), _mm256_andnot_ps(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sign_bit_select, (batch_t sign_mask, batch_t a, batch_t b))
    {
        return { _mm256_blendv_ps(b.v, a.v, sign_mask.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, lane_select, (batch_t lane_mask, batch_t a, batch_t b))
    {
        __m256 mask = _mm256_cmp_ps(lane_mask.v, _mm256_setzero_ps(), _CMP_NEQ_UQ); // UQ: NaN -> one_block, select a
        return { _mm256_or_ps(_mm256_and_ps(mask, a.v), _mm256_andnot_ps(mask, b.v)) };
    }
};

// AVX2 + FMA指令特化
template<>
struct SimdOp<SimdInstruction::AVX2_FMA3_F16C, float32>
    : SimdOp<SimdInstruction::AVX2, float32>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::AVX2_FMA3_F16C, float32)

    KSIMD_OP_SIG_AVX2_FMA3_F16C(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm256_fmadd_ps(a.v, b.v, c.v) };
    }
};

KSIMD_NAMESPACE_END

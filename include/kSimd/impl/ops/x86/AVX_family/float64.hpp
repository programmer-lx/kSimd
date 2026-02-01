#pragma once

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I>
    requires (I >= SimdInstruction::AVX && I <= SimdInstruction::AVX2)
struct SimdOp<I, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::AVX, float64)

    KSIMD_OP_SIG_AVX(mask_t, mask_from_lanes, (unsigned int count))
    {
        __m256d idx = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d cnt = _mm256_set1_pd(static_cast<float64>(count));
        return { _mm256_cmp_pd(idx, cnt, _CMP_LT_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, load, (const float64* mem))
    {
        return { _mm256_load_pd(mem) };
    }

    KSIMD_OP_SIG_AVX(batch_t, loadu, (const float64* mem))
    {
        return { _mm256_loadu_pd(mem) };
    }

    KSIMD_OP_SIG_AVX(void, store, (float64* mem, batch_t v))
    {
        _mm256_store_pd(mem, v.v);
    }

    KSIMD_OP_SIG_AVX(void, storeu, (float64* mem, batch_t v))
    {
        _mm256_storeu_pd(mem, v.v);
    }

    KSIMD_OP_SIG_AVX(batch_t, load_masked, (const float64* mem, mask_t mask))
    {
        uint32 m = _mm256_movemask_pd(mask.m); // [3:0]有效
        alignas(BatchAlignment) float64 tmp[Lanes]{};
        for (size_t i = 0; i < Lanes; ++i)
        {
            if (m & (1 << i))
            {
                tmp[i] = mem[i];
            }
            else
            {
                tmp[i] = 0.0;
            }
        }
        return { _mm256_load_pd(tmp) };
    }

    KSIMD_OP_SIG_AVX(batch_t, zero, ())
    {
        return { _mm256_setzero_pd() };
    }

    KSIMD_OP_SIG_AVX(batch_t, set, (float64 x))
    {
        return { _mm256_set1_pd(x) };
    }

    KSIMD_OP_SIG_AVX(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, one_div, (batch_t v))
    {
        return { _mm256_div_pd(_mm256_set1_pd(1.0), v.v) };
    }

    KSIMD_OP_SIG_AVX(float64, reduce_sum, (batch_t v))
    {
        // [4, 3, 2, 1]
        // hadd
        // [3+4, 3+4, 1+2, 1+2]
        __m256d t1 = _mm256_hadd_pd(v.v, v.v);

        // [1+2, 1+2] low
        //     +
        // [3+4, 3+4] high
        __m128d low = _mm256_castpd256_pd128(t1);
        __m128d high = _mm256_extractf128_pd(t1, 0b1);
        high = _mm_add_pd(low, high);

        // get lane[0]
        return _mm_cvtsd_f64(high);
    }

    KSIMD_OP_SIG_AVX(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sqrt, (batch_t v))
    {
        return { _mm256_sqrt_pd(v.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, rsqrt, (batch_t v))
    {
        return { _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(v.v)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, abs, (batch_t v))
    {
        return { _mm256_and_pd(v.v, _mm256_set1_pd(sign_bit_clear_mask<float64>)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_min_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_max_pd(lhs.v, rhs.v) };
    }
    
    KSIMD_OP_SIG_AVX(batch_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    KSIMD_OP_SIG_AVX(batch_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    KSIMD_OP_SIG_AVX(batch_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_ORD_Q) };
    }
    
    KSIMD_OP_SIG_AVX(batch_t, bit_not, (batch_t v))
    {
        return { _mm256_xor_pd(v.v, _mm256_set1_pd(one_block<float64>)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_andnot_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm256_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm256_or_pd(_mm256_and_pd(mask.v, a.v), _mm256_andnot_pd(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_AVX(batch_t, sign_bit_select, (batch_t sign_mask, batch_t a, batch_t b))
    {
        return { _mm256_blendv_pd(b.v, a.v, sign_mask.v) };
    }

    KSIMD_OP_SIG_AVX(batch_t, lane_select, (batch_t lane_mask, batch_t a, batch_t b))
    {
        __m256d mask = _mm256_cmp_pd(lane_mask.v, _mm256_setzero_pd(), _CMP_NEQ_UQ); // UQ: NaN -> one_block, select a
        return { _mm256_or_pd(_mm256_and_pd(mask, a.v), _mm256_andnot_pd(mask, b.v)) };
    }
};

template<>
struct SimdOp<SimdInstruction::AVX2_FMA3_F16C, float64> : SimdOp<SimdInstruction::AVX2, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::AVX2_FMA3_F16C, float64)

    KSIMD_OP_SIG_AVX2_FMA3_F16C(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm256_fmadd_pd(a.v, b.v, c.v) };
    }
};

KSIMD_NAMESPACE_END

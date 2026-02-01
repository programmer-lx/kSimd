#pragma once

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE, float64>
    : detail::SimdOp_Scalar_FloatingPoint_Base<SimdInstruction::SSE, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE, float64)
};

template<>
struct SimdOp<SimdInstruction::SSE2, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE2, float64)

    KSIMD_OP_SIG_SSE(mask_t, mask_from_lanes, (unsigned int count))
    {
        __m128d idx = _mm_set_pd(1.0, 0.0);
        __m128d cnt = _mm_set1_pd(static_cast<float64>(count));
        return { _mm_cmplt_pd(idx, cnt) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, load, (const float64* mem))
    {
        return { _mm_load_pd(mem) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, loadu, (const float64* mem))
    {
        return { _mm_loadu_pd(mem) };
    }

    KSIMD_OP_SIG_SSE2(void, store, (float64* mem, batch_t v))
    {
        _mm_store_pd(mem, v.v);
    }

    KSIMD_OP_SIG_SSE2(void, storeu, (float64* mem, batch_t v))
    {
        _mm_storeu_pd(mem, v.v);
    }

    KSIMD_OP_SIG_SSE2(batch_t, load_masked, (const float64* mem, mask_t mask))
    {
        uint32 m = _mm_movemask_pd(mask.m); // 仅 [3:0] 有效
        alignas(BatchAlignment) float64 tmp[Lanes]{};
        for (size_t i = 0; i < Lanes; ++i)
        {
            if (m & (1 << i))
            {
                tmp[i] = mem[i];
            }
            else
            {
                tmp[i] = 0.0f;
            }
        }
        return { _mm_load_pd(tmp) };
    }

   KSIMD_OP_SIG_SSE2(batch_t, zero, ())
    {
        return { _mm_setzero_pd() };
    }

    KSIMD_OP_SIG_SSE2(batch_t, set, (float64 x))
    {
        return { _mm_set1_pd(x) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, one_div, (batch_t v))
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), v.v) };
    }

    KSIMD_OP_SIG_SSE2(float64, reduce_sum, (batch_t v))
    {
        // [b, a]
        //   +
        // [a, b]
        //   =
        // [a+b]
        // get lane[0]

        __m128d t1 = _mm_shuffle_pd(v.v, v.v, _MM_SHUFFLE2(0, 1));
        t1 = _mm_add_pd(v.v, t1);
        return _mm_cvtsd_f64(t1);
    }

    KSIMD_OP_SIG_SSE2(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm_add_pd(_mm_mul_pd(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, sqrt, (batch_t v))
    {
        return { _mm_sqrt_pd(v.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, rsqrt, (batch_t v))
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v.v)) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, abs, (batch_t v))
    {
        return { _mm_and_pd(v.v, _mm_set1_pd(sign_bit_clear_mask<float64>)) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm_min_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm_max_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpeq_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpneq_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpgt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpngt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpge_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnge_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmplt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnlt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmple_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnle_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpunord_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpord_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_not, (batch_t v))
    {
        return { _mm_xor_pd(v.v, _mm_set1_pd(one_block<float64>)) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm_andnot_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_pd(_mm_and_pd(mask.v, a.v), _mm_andnot_pd(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, sign_bit_select, (batch_t sign_mask, batch_t a, batch_t b))
    {
        // 直接读取sign bit，构造mask，然后select
        __m128i sign_mask_i64 = _mm_castpd_si128(sign_mask.v);
        __m128i mask_i = _mm_srai_epi32(sign_mask_i64, 31);
        __m128d mask = _mm_castsi128_pd(mask_i);

        return { _mm_or_pd(_mm_and_pd(mask, a.v), _mm_andnot_pd(mask, b.v)) };
    }

    KSIMD_OP_SIG_SSE2(batch_t, lane_select, (batch_t lane_mask, batch_t a, batch_t b))
    {
        __m128d mask = _mm_cmpneq_pd(lane_mask.v, _mm_setzero_pd());
        return { _mm_or_pd(_mm_and_pd(mask, a.v), _mm_andnot_pd(mask, b.v)) };
    }
};

template<SimdInstruction I>
    requires (I >= SimdInstruction::SSE3 && I <= SimdInstruction::SSE4_1)
struct SimdOp<I, float64>
    : SimdOp<SimdInstruction::SSE2, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE3, float64)

    KSIMD_OP_SIG_SSE3(float64, reduce_sum, (batch_t v))
    {
        // input: [b, a]
        // hadd: [a+b]
        // get lane[0]
        __m128d result = _mm_hadd_pd(v.v, v.v);
        return _mm_cvtsd_f64(result);
    }
};

KSIMD_NAMESPACE_END

#pragma once

#include <bit>

#include "../_SSE_family_types.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct SimdOp<SimdInstruction::SSE, float32>
{
    using traits = SimdTraits<SimdInstruction::SSE, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;

    KSIMD_OP_SIG_SSE(batch_t, load, (const float32* mem))
    {
        return { _mm_load_ps(mem) };
    }

    KSIMD_OP_SIG_SSE(batch_t, loadu, (const float32* mem))
    {
        return { _mm_loadu_ps(mem) };
    }

    KSIMD_OP_SIG_SSE(void, store, (float32* mem, batch_t v))
    {
        _mm_store_ps(mem, v.v);
    }

    KSIMD_OP_SIG_SSE(void, storeu, (float32* mem, batch_t v))
    {
        _mm_storeu_ps(mem, v.v);
    }

   KSIMD_OP_SIG_SSE(batch_t, zero, ())
    {
        return { _mm_setzero_ps() };
    }

    KSIMD_OP_SIG_SSE(batch_t, set, (float32 x))
    {
        return { _mm_set1_ps(x) };
    }

    KSIMD_OP_SIG_SSE(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, one_div, (batch_t v))
    {
        return { _mm_rcp_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE(float32, reduce_sum, (batch_t v))
    {
        // [d, c, b, a]
        //       +
        // [c, d, a, b]
        //       =
        // [c+d, c+d, a+b, a+b]
        //       +
        // [a+b, a+b, c+d, c+d]
        // [a+b+c+d, ...]
        // get lane[0]

        __m128 t1 = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(2, 3, 0, 1));
        t1 = _mm_add_ps(v.v, t1);
        v.v = _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 0, 3, 2));
        t1 = _mm_add_ps(t1, v.v);
        return _mm_cvtss_f32(t1);
    }

    KSIMD_OP_SIG_SSE(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, sqrt, (batch_t v))
    {
        return { _mm_sqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, rsqrt, (batch_t v))
    {
        return { _mm_rsqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, abs, (batch_t v))
    {
        return { _mm_and_ps(v.v, _mm_set1_ps(sign_bit_clear_mask<float>)) };
    }

    KSIMD_OP_SIG_SSE(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm_min_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm_max_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, clamp, (batch_t v, batch_t range1, batch_t range2))
    {
        __m128 min = _mm_min_ps(range1.v, range2.v);
        __m128 max = _mm_max_ps(range1.v, range2.v);
        return { _mm_min_ps(_mm_max_ps(v.v, min), max) };
    }

    KSIMD_OP_SIG_SSE(batch_t, unsafe_clamp, (batch_t v, batch_t min, batch_t max))
    {
        return { _mm_min_ps(_mm_max_ps(v.v, min.v), max.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, lerp, (batch_t a, batch_t b, batch_t t))
    {
        __m128 b_a = _mm_sub_ps(b.v, a.v);
        return { _mm_add_ps(a.v, _mm_mul_ps(b_a, t.v)) };
    }

    KSIMD_OP_SIG_SSE(batch_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpeq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpneq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpgt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpngt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmplt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnlt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmple_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnle_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpunord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, bit_not, (batch_t v))
    {
        return { _mm_xor_ps(v.v, _mm_set1_ps(one_block<float>)) };
    }
    
    KSIMD_OP_SIG_SSE(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm_andnot_ps(lhs.v, rhs.v) };
    }
    
    KSIMD_OP_SIG_SSE(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm_or_ps(lhs.v, rhs.v) };
    }
    
    KSIMD_OP_SIG_SSE(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_ps(_mm_and_ps(mask.v, a.v), _mm_andnot_ps(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_SSE(batch_t, sign_bit_select, (batch_t sign_mask, batch_t a, batch_t b))
    {
        // 直接读取sign bit，然后构造mask
        alignas(Alignment::SSE_Family) float32 tmp[traits::Lanes];
        _mm_store_ps(tmp, sign_mask.v);

        constexpr uint32 sign_bit = sign_bit_mask<uint32>;
        for (size_t i = 0; i < traits::Lanes; ++i)
        {
            tmp[i] = (std::bit_cast<uint32>(tmp[i]) & sign_bit) ? one_block<float32> : zero_block<float32>;
        }

        __m128 mask = _mm_load_ps(tmp);
        return { _mm_or_ps(_mm_and_ps(mask, a.v), _mm_andnot_ps(mask, b.v)) };
    }

    KSIMD_OP_SIG_SSE(batch_t, lane_select, (batch_t lane_mask, batch_t a, batch_t b))
    {
        // 使用 bit_select 方案，但是需要使用 cmp_neq 来构造 mask
        __m128 mask = _mm_cmpneq_ps(lane_mask.v, _mm_setzero_ps());
        return { _mm_or_ps(_mm_and_ps(mask, a.v), _mm_andnot_ps(mask, b.v)) };
    }
};

KSIMD_NAMESPACE_END

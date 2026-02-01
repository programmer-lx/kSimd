#pragma once

#include <bit>

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I>
    requires (I >= SimdInstruction::SSE && I <= SimdInstruction::SSE2)
struct SimdOp<I, float32>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE, float32)

    #if defined(KSIMD_IS_TESTING)
    KSIMD_OP_SIG_SSE(void, test_store_mask, (float32* mem, mask_t mask))
    {
        _mm_store_ps(mem, mask.m);
    }
    #endif

    KSIMD_OP_SIG_SSE(mask_t, mask_from_lanes, (unsigned int count))
    {
        __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
        __m128 cnt = _mm_set1_ps(static_cast<float32>(count));
        return { _mm_cmplt_ps(idx, cnt) };
    }

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

    KSIMD_OP_SIG_SSE(batch_t, mask_load, (const float32* mem, mask_t mask))
    {
        uint32 m = _mm_movemask_ps(mask.m); // 仅 [3:0] 有效
        alignas(BatchAlignment) float32 tmp[Lanes]{};
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
        return { _mm_load_ps(tmp) };
    }

    KSIMD_OP_SIG_SSE(batch_t, undefined, ())
    {
        return { _mm_undefined_ps() };
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
        return { _mm_and_ps(v.v, _mm_set1_ps(sign_bit_clear_mask<float32>)) };
    }

    KSIMD_OP_SIG_SSE(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm_min_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm_max_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpeq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpneq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpgt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpngt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmplt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnlt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmple_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnle_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpunord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(mask_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(batch_t, bit_not, (batch_t v))
    {
        return { _mm_xor_ps(v.v, _mm_set1_ps(one_block<float32>)) };
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
};

template<SimdInstruction I>
    requires (I >= SimdInstruction::SSE3 && I <= SimdInstruction::SSE4_1)
struct SimdOp<I, float32>
    : SimdOp<SimdInstruction::SSE2, float32>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE3, float32)

    KSIMD_OP_SIG_SSE3(float32, reduce_sum, (batch_t v))
    {
        // input: [d, c, b, a]
        // hadd: [c+d, a+b, c+d, a+b]
        // hadd: [a+b+c+d, .........]
        // get lane[0]

        __m128 result = _mm_hadd_ps(v.v, v.v);
        result = _mm_hadd_ps(result, result);
        return _mm_cvtss_f32(result);
    }
};

KSIMD_NAMESPACE_END

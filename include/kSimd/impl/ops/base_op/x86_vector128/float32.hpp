#pragma once

#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/utils.hpp"
#include "kSimd/impl/func_attr.hpp"

KSIMD_NAMESPACE_BEGIN

// -------------------------------- operators --------------------------------
namespace x86_vector128
{
    #define KSIMD_BATCH_T Batch<float32>

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator+, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator-, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator*, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator/, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator-, (KSIMD_BATCH_T v))
    {
        return { _mm_sub_ps(_mm_setzero_ps(), v.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator&, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator|, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator^, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator~, (KSIMD_BATCH_T v))
    {
        return { _mm_xor_ps(v.v, _mm_set1_ps(OneBlock<float32>)) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator+=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs + rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator-=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs - rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator*=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs * rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator/=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs / rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator&=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs & rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator|=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs | rhs;
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T&, operator^=, (KSIMD_BATCH_T& lhs, KSIMD_BATCH_T rhs))
    {
        return lhs = lhs ^ rhs;
    }

    #undef KSIMD_BATCH_T
}

template<>
struct BaseOp<SimdInstruction::SSE, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::SSE, float32)

    #if defined(KSIMD_IS_TESTING)
    KSIMD_OP_SIG_SSE_STATIC(void, test_store_mask, (float32* mem, mask_t mask))
    {
        _mm_store_ps(mem, mask.m);
    }
    KSIMD_OP_SIG_SSE_STATIC(mask_t, test_load_mask, (const float32* mem))
    {
        return { _mm_load_ps(mem) };
    }
    #endif

    KSIMD_OP_SIG_SSE_STATIC(mask_t, mask_from_lanes, (unsigned int count))
    {
        __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
        __m128 cnt = _mm_set1_ps(static_cast<float32>(count));
        return { _mm_cmplt_ps(idx, cnt) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, load, (const float32* mem))
    {
        return { _mm_load_ps(mem) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, loadu, (const float32* mem))
    {
        return { _mm_loadu_ps(mem) };
    }

    KSIMD_OP_SIG_SSE_STATIC(void, store, (float32* mem, batch_t v))
    {
        _mm_store_ps(mem, v.v);
    }

    KSIMD_OP_SIG_SSE_STATIC(void, storeu, (float32* mem, batch_t v))
    {
        _mm_storeu_ps(mem, v.v);
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, mask_load, (const float32* mem, mask_t mask))
    {
        __m128 lane0 = _mm_setzero_ps();
        __m128 lane1 = _mm_setzero_ps();
        __m128 lane2 = _mm_setzero_ps();
        __m128 lane3 = _mm_setzero_ps();

        const uint32 m = _mm_movemask_ps(mask.m); // [3:0] 有效

        if (m & 0b0001)
        {
            // [ 0, 0, 0, mem[0] ]
            lane0 = _mm_load_ss(mem);
        }
        if (m & 0b0010)
        {
            // [ 0, 0, 0, mem[1] ]
            lane1 = _mm_load_ss(mem + 1);
        }
        if (m & 0b0100)
        {
            // [ 0, 0, 0, mem[2] ]
            lane2 = _mm_load_ss(mem + 2);
        }
        if (m & 0b1000)
        {
            // [ 0, 0, 0, mem[3] ]
            lane3 = _mm_load_ss(mem + 3);
        }

        // lane0 + lane1 = [ 0, 0, mem[1], mem[0] ]
        __m128 lane01 = _mm_unpacklo_ps(lane0, lane1);

        // lane2 + lane3 = [ 0, 0, mem[3], mem[2] ]
        __m128 lane23 = _mm_unpacklo_ps(lane2, lane3);

        return { _mm_movelh_ps(lane01, lane23) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, mask_loadu, (const float32* mem, mask_t mask))
    {
        __m128 lane0 = _mm_setzero_ps();
        __m128 lane1 = _mm_setzero_ps();
        __m128 lane2 = _mm_setzero_ps();
        __m128 lane3 = _mm_setzero_ps();

        const uint32 m = _mm_movemask_ps(mask.m); // [3:0] 有效

        if (m & 0b0001)
        {
            // [ 0, 0, 0, mem[0] ]
            lane0 = _mm_load_ss(mem);
        }
        if (m & 0b0010)
        {
            // [ 0, 0, 0, mem[1] ]
            lane1 = _mm_load_ss(mem + 1);
        }
        if (m & 0b0100)
        {
            // [ 0, 0, 0, mem[2] ]
            lane2 = _mm_load_ss(mem + 2);
        }
        if (m & 0b1000)
        {
            // [ 0, 0, 0, mem[3] ]
            lane3 = _mm_load_ss(mem + 3);
        }

        // lane0 + lane1 = [ 0, 0, mem[1], mem[0] ]
        __m128 lane01 = _mm_unpacklo_ps(lane0, lane1);

        // lane2 + lane3 = [ 0, 0, mem[3], mem[2] ]
        __m128 lane23 = _mm_unpacklo_ps(lane2, lane3);

        return { _mm_movelh_ps(lane01, lane23) };
    }

    KSIMD_OP_SIG_SSE_STATIC(void, mask_store, (float32* mem, batch_t v, mask_t mask))
    {
        const uint32_t m = _mm_movemask_ps(mask.m); // [3:0]有效

        if (m & 0b0001)
        {
            _mm_store_ss(mem, v.v);
        }
        if (m & 0b0010)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(1, 1, 1, 1));
            _mm_store_ss(mem + 1, tmp);
        }
        if (m & 0b0100)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(2, 2, 2, 2));
            _mm_store_ss(mem + 2, tmp);
        }
        if (m & 0b1000)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(3, 3, 3, 3));
            _mm_store_ss(mem + 3, tmp);
        }
    }

    KSIMD_OP_SIG_SSE_STATIC(void, mask_storeu, (float32* mem, batch_t v, mask_t mask))
    {
        const uint32_t m = _mm_movemask_ps(mask.m); // [3:0]有效

        if (m & 0b0001)
        {
            _mm_store_ss(mem, v.v);
        }
        if (m & 0b0010)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(1, 1, 1, 1));
            _mm_store_ss(mem + 1, tmp);
        }
        if (m & 0b0100)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(2, 2, 2, 2));
            _mm_store_ss(mem + 2, tmp);
        }
        if (m & 0b1000)
        {
            __m128 tmp = _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(3, 3, 3, 3));
            _mm_store_ss(mem + 3, tmp);
        }
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, undefined, ())
    {
        return { _mm_undefined_ps() };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, zero, ())
    {
        return { _mm_setzero_ps() };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, set, (float32 x))
    {
        return { _mm_set1_ps(x) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, one_div, (batch_t v))
    {
        return { _mm_rcp_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(float32, reduce_add, (batch_t v))
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

    KSIMD_OP_SIG_SSE_STATIC(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, sqrt, (batch_t v))
    {
        return { _mm_sqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, rsqrt, (batch_t v))
    {
        return { _mm_rsqrt_ps(v.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, abs, (batch_t v))
    {
        return { _mm_and_ps(v.v, _mm_set1_ps(SignBitClearMask<float32>)) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm_min_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm_max_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpeq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpneq_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpgt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpngt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnge_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmplt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnlt_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmple_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnle_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpunord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, all_NaN, (batch_t lhs, batch_t rhs))
    {
        __m128 l_nan = _mm_cmpunord_ps(lhs.v, lhs.v);
        __m128 r_nan = _mm_cmpunord_ps(rhs.v, rhs.v);
        return { _mm_and_ps(l_nan, r_nan) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpord_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, any_finite, (batch_t lhs, batch_t rhs))
    {
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float32>);
        __m128 inf = _mm_set1_ps(Inf<float32>);

        // 如果一个是有限值(指数位有0)，AND 之后结果的指数位一定会有0
        __m128 combined = _mm_and_ps(lhs.v, rhs.v);

        return { _mm_cmplt_ps(_mm_and_ps(combined, abs_mask), inf) };
    }

    KSIMD_OP_SIG_SSE_STATIC(mask_t, all_finite, (batch_t lhs, batch_t rhs))
    {
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float32>);
        __m128 inf = _mm_set1_ps(Inf<float32>);

        __m128 l_finite = _mm_cmplt_ps(_mm_and_ps(lhs.v, abs_mask), inf);
        __m128 r_finite = _mm_cmplt_ps(_mm_and_ps(rhs.v, abs_mask), inf);

        return { _mm_and_ps(l_finite, r_finite) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_not, (batch_t v))
    {
        return { _mm_xor_ps(v.v, _mm_set1_ps(OneBlock<float32>)) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm_andnot_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_ps(_mm_and_ps(mask.v, a.v), _mm_andnot_ps(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_SSE_STATIC(batch_t, mask_select, (mask_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_ps(_mm_and_ps(mask.m, a.v), _mm_andnot_ps(mask.m, b.v)) };
    }
};

template<>
struct BaseOp<SimdInstruction::SSE2, float32> : BaseOp<SimdInstruction::SSE, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::SSE2, float32)
};

template<>
struct BaseOp<SimdInstruction::SSE3, float32> : BaseOp<SimdInstruction::SSE2, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::SSE3, float32)

    KSIMD_OP_SIG_SSE3_STATIC(float32, reduce_add, (batch_t v))
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

template<>
struct BaseOp<SimdInstruction::SSE4_1, float32> : BaseOp<SimdInstruction::SSE3, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::SSE4_1, float32)

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, mask_select, (mask_t mask, batch_t a, batch_t b))
    {
        return { _mm_blendv_ps(b.v, a.v, mask.m) };
    }

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, round_up, (batch_t v))
    {
        return { _mm_round_ps(v.v, _MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC) };
    }

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, round_down, (batch_t v))
    {
        return { _mm_round_ps(v.v, _MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC) };
    }

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, round_nearest, (batch_t v))
    {
        return { _mm_round_ps(v.v, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) };
    }

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, round, (batch_t v))
    {
        // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
        __m128 sign_mask = _mm_and_ps(v.v, _mm_set1_ps(SignBitMask<float32>));

        // 构造一个具有相同符号的0.5，但是比0.5小一点，防止进位
        __m128 half = _mm_or_ps(_mm_set1_ps(0.49999997f), sign_mask);

        return { _mm_round_ps(_mm_add_ps(v.v, half), _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC) };
    }

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, round_to_zero, (batch_t v))
    {
        return { _mm_round_ps(v.v, _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC) };
    }
};

KSIMD_NAMESPACE_END

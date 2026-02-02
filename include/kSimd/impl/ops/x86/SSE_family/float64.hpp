#pragma once

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

// -------------------------------- operators --------------------------------
namespace SSE_family::SSE2_up
{
    #define KSIMD_BATCH_T Batch<float64>

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator+, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator-, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator*, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator/, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator-, (KSIMD_BATCH_T v))
    {
        return { _mm_sub_pd(_mm_setzero_pd(), v.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator&, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator|, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator^, (KSIMD_BATCH_T lhs, KSIMD_BATCH_T rhs))
    {
        return { _mm_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE(KSIMD_BATCH_T, operator~, (KSIMD_BATCH_T v))
    {
        return { _mm_xor_pd(v.v, _mm_set1_pd(one_block<float64>)) };
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


#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
template<>
struct SimdOp<SimdInstruction::SSE, float64>
    : detail::SimdOp_Scalar_FloatingPoint_Base<SimdInstruction::SSE, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE, float64)
};
#endif


template<>
struct SimdOp<SimdInstruction::SSE2, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE2, float64)
    
    #if defined(KSIMD_IS_TESTING)
    KSIMD_OP_SIG_SSE2_STATIC(void, test_store_mask, (float64* mem, mask_t mask))
    {
        _mm_store_pd(mem, mask.m);
    }
    KSIMD_OP_SIG_SSE2_STATIC(mask_t, test_load_mask, (const float64* mem))
    {
        return { _mm_load_pd(mem) };
    }
    #endif

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, mask_from_lanes, (unsigned int count))
    {
        __m128d idx = _mm_set_pd(1.0, 0.0);
        __m128d cnt = _mm_set1_pd(static_cast<float64>(count));
        return { _mm_cmplt_pd(idx, cnt) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, load, (const float64* mem))
    {
        return { _mm_load_pd(mem) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, loadu, (const float64* mem))
    {
        return { _mm_loadu_pd(mem) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(void, store, (float64* mem, batch_t v))
    {
        _mm_store_pd(mem, v.v);
    }

    KSIMD_OP_SIG_SSE2_STATIC(void, storeu, (float64* mem, batch_t v))
    {
        _mm_storeu_pd(mem, v.v);
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, mask_load, (const float64* mem, mask_t mask))
    {
        uint32 m = _mm_movemask_pd(mask.m); // 仅 [3:0] 有效
        alignas(BatchAlignment) float64 tmp[Lanes]{};

        [&]<size_t... I>(std::index_sequence<I...>)
        {
            ((tmp[I] = (m & (1 << I)) ? mem[I] : 0.0), ...);
        }(std::make_index_sequence<Lanes>{});

        return { _mm_load_pd(tmp) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, mask_loadu, (const float64* mem, mask_t mask))
    {
        uint32 m = _mm_movemask_pd(mask.m); // 仅 [3:0] 有效
        alignas(BatchAlignment) float64 tmp[Lanes]{};

        [&]<size_t... I>(std::index_sequence<I...>)
        {
            ((tmp[I] = (m & (1 << I)) ? mem[I] : 0.0), ...);
        }(std::make_index_sequence<Lanes>{});

        return { _mm_load_pd(tmp) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(void, mask_store, (float64* mem, batch_t v, mask_t mask))
    {
        alignas(BatchAlignment) float64 tmp[Lanes]{};
        _mm_store_pd(tmp, v.v);

        const uint32_t m = _mm_movemask_pd(mask.m); // [1:0]有效
        [&]<size_t... I>(std::index_sequence<I...>)
        {
            ( ((m & (1 << I)) ? (mem[I] = tmp[I], void()) : void()), ... );
        }(std::make_index_sequence<Lanes>{});
    }

    KSIMD_OP_SIG_SSE2_STATIC(void, mask_storeu, (float64* mem, batch_t v, mask_t mask))
    {
        alignas(BatchAlignment) float64 tmp[Lanes]{};
        _mm_store_pd(tmp, v.v);

        const uint32_t m = _mm_movemask_pd(mask.m); // [1:0]有效
        [&]<size_t... I>(std::index_sequence<I...>)
        {
            ( ((m & (1 << I)) ? (mem[I] = tmp[I], void()) : void()), ... );
        }(std::make_index_sequence<Lanes>{});
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, undefined, ())
    {
        return { _mm_undefined_pd() };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, zero, ())
    {
        return { _mm_setzero_pd() };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, set, (float64 x))
    {
        return { _mm_set1_pd(x) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, add, (batch_t lhs, batch_t rhs))
    {
        return { _mm_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, sub, (batch_t lhs, batch_t rhs))
    {
        return { _mm_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, mul, (batch_t lhs, batch_t rhs))
    {
        return { _mm_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, div, (batch_t lhs, batch_t rhs))
    {
        return { _mm_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, one_div, (batch_t v))
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), v.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(float64, reduce_sum, (batch_t v))
    {
        // [b, a]
        //   +
        // [a, b]
        //   =
        // [a+b]
        // get lane[0]

        __m128d sum64 = _mm_add_pd(v.v, _mm_shuffle_pd(v.v, v.v, _MM_SHUFFLE2(0, 1)));
        return _mm_cvtsd_f64(sum64);
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
    {
        return { _mm_add_pd(_mm_mul_pd(a.v, b.v), c.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, sqrt, (batch_t v))
    {
        return { _mm_sqrt_pd(v.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, rsqrt, (batch_t v))
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v.v)) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, abs, (batch_t v))
    {
        return { _mm_and_pd(v.v, _mm_set1_pd(sign_bit_clear_mask<float64>)) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, min, (batch_t lhs, batch_t rhs))
    {
        return { _mm_min_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, max, (batch_t lhs, batch_t rhs))
    {
        return { _mm_max_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpeq_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpneq_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpgt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_greater, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpngt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpge_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_greater_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnge_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmplt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_less, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnlt_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmple_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_less_equal, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpnle_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, any_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpunord_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(mask_t, not_NaN, (batch_t lhs, batch_t rhs))
    {
        return { _mm_cmpord_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_not, (batch_t v))
    {
        return { _mm_xor_pd(v.v, _mm_set1_pd(one_block<float64>)) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_and, (batch_t lhs, batch_t rhs))
    {
        return { _mm_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
    {
        return { _mm_andnot_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_or, (batch_t lhs, batch_t rhs))
    {
        return { _mm_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
    {
        return { _mm_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_pd(_mm_and_pd(mask.v, a.v), _mm_andnot_pd(mask.v, b.v)) };
    }

    KSIMD_OP_SIG_SSE2_STATIC(batch_t, mask_select, (mask_t mask, batch_t a, batch_t b))
    {
        return { _mm_or_pd(_mm_and_pd(mask.m, a.v), _mm_andnot_pd(mask.m, b.v)) };
    }
};

template<>
struct SimdOp<SimdInstruction::SSE3, float64> : SimdOp<SimdInstruction::SSE2, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE3, float64)

    KSIMD_OP_SIG_SSE3_STATIC(float64, reduce_sum, (batch_t v))
    {
        // input: [b, a]
        // hadd: [a+b]
        // get lane[0]
        __m128d result = _mm_hadd_pd(v.v, v.v);
        return _mm_cvtsd_f64(result);
    }
};

template<>
struct SimdOp<SimdInstruction::SSE4_1, float64> : SimdOp<SimdInstruction::SSE3, float64>
{
    KSIMD_DETAIL_SIMD_OP_TRAITS(SimdInstruction::SSE4_1, float64)

    KSIMD_OP_SIG_SSE4_1_STATIC(batch_t, mask_select, (mask_t mask, batch_t a, batch_t b))
    {
        return { _mm_blendv_pd(b.v, a.v, mask.m) };
    }
};

KSIMD_NAMESPACE_END

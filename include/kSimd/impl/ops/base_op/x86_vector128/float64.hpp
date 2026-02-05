#pragma once

#include "traits.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 1.0, 0.0

KSIMD_NAMESPACE_BEGIN

// -------------------------------- operators --------------------------------
#define KSIMD_API(...) KSIMD_OP_SSE2_API static __VA_ARGS__ KSIMD_CALL_CONV
namespace x86_vector128
{
#define KSIMD_BATCH_T Batch<float64, reg_count>

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator+(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_add_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_sub_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator*(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_mul_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator/(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_div_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> v) noexcept
    {
        return { _mm_xor_pd(v.v[0], _mm_set1_pd(SignBitMask<float64>)) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator&(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_and_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator|(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_or_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator^(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return { _mm_xor_pd(lhs.v[0], rhs.v[0]) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator~(Batch<float64, reg_count> v) noexcept
    {
        return { _mm_xor_pd(v.v[0], _mm_set1_pd(OneBlock<float64>)) };
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator+=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator-=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator*=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator/=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator&=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator|=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>&) operator^=(Batch<float64, reg_count>& lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }

#undef KSIMD_BATCH_T
} // namespace x86_vector128
#undef KSIMD_API


// SSE float64 使用标量模拟
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float64>
    : detail::BaseOp_Scalar_FloatingPoint_Base<
        SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE,
        vector_scalar::Batch<float64, 2, alignof(float64)>,
        vector_scalar::Mask<float64, 2, alignof(float64)>, alignof(float64)
    >
{
    KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE<float64>)
};
#endif


#define KSIMD_API(ret) KSIMD_OP_SSE2_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float64>
{
    KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE2_Plus<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float64, 1>)

#if defined(KSIMD_IS_TESTING)
    KSIMD_API(void) test_store_mask(float64* mem, mask_t mask) noexcept
    {
        _mm_store_pd(mem, mask.m[0]);
    }
    KSIMD_API(mask_t) test_load_mask(const float64* mem) noexcept
    {
        return { _mm_load_pd(mem) };
    }
#endif

    KSIMD_API(mask_t) mask_from_lanes(size_t count) noexcept
    {
        __m128d idx = _mm_set_pd(KSIMD_IOTA);
        __m128d cnt = _mm_set1_pd(static_cast<float64>(count));
        return { _mm_cmplt_pd(idx, cnt) };
    }

    KSIMD_API(batch_t) load(const float64* mem) noexcept
    {
        return { _mm_load_pd(mem) };
    }

    KSIMD_API(batch_t) loadu(const float64* mem) noexcept
    {
        return { _mm_loadu_pd(mem) };
    }

    KSIMD_API(void) store(float64* mem, batch_t v) noexcept
    {
        _mm_store_pd(mem, v.v[0]);
    }

    KSIMD_API(void) storeu(float64* mem, batch_t v) noexcept
    {
        _mm_storeu_pd(mem, v.v[0]);
    }

    KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask) noexcept
    {
        __m128d result = _mm_setzero_pd();

        const int32 m = _mm_movemask_pd(mask.m[0]); // 仅 [1:0] 有效
        if (m & 0b01)
            result = _mm_load_sd(mem);
        if (m & 0b10)
            result = _mm_loadh_pd(result, mem + 1);
        return { result };
    }

    KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask, batch_t default_value) noexcept
    {
        batch_t loaded = mask_load(mem, mask);
        return { _mm_or_pd(loaded.v[0], _mm_andnot_pd(mask.m[0], default_value.v[0])) };
    }

    KSIMD_API(batch_t) mask_loadu(const float64* mem, mask_t mask) noexcept
    {
        return mask_load(mem, mask);
    }

    KSIMD_API(batch_t) mask_loadu(const float64* mem, mask_t mask, batch_t default_value) noexcept
    {
        return mask_load(mem, mask, default_value);
    }

    KSIMD_API(void) mask_store(float64* mem, batch_t v, mask_t mask) noexcept
    {
        const int32 m = _mm_movemask_pd(mask.m[0]); // [1:0]有效
        if (m & 0b01)
            _mm_store_sd(mem, v.v[0]);
        if (m & 0b10)
            _mm_storeh_pd(mem + 1, v.v[0]);
    }

    KSIMD_API(void) mask_storeu(float64* mem, batch_t v, mask_t mask) noexcept
    {
        const int32 m = _mm_movemask_pd(mask.m[0]); // [1:0]有效
        if (m & 0b01)
            _mm_store_sd(mem, v.v[0]);
        if (m & 0b10)
            _mm_storeh_pd(mem + 1, v.v[0]);
    }

    KSIMD_API(batch_t) undefined() noexcept
    {
        return { _mm_undefined_pd() };
    }

    KSIMD_API(batch_t) zero() noexcept
    {
        return { _mm_setzero_pd() };
    }

    KSIMD_API(batch_t) set(float64 x) noexcept
    {
        return { _mm_set1_pd(x) };
    }
    
    KSIMD_API(batch_t) sequence() noexcept
    {
        return { _mm_set_pd(KSIMD_IOTA) };
    }

    KSIMD_API(batch_t) sequence(float64 base) noexcept
    {
        __m128d iota = _mm_set_pd(KSIMD_IOTA);
        return { _mm_add_pd(iota, _mm_set1_pd(base)) };
    }

    KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
    {
        __m128d iota = _mm_set_pd(KSIMD_IOTA);
        __m128d stride_v = _mm_set1_pd(stride);
        __m128d base_v = _mm_set1_pd(base);
        return { _mm_add_pd(_mm_mul_pd(stride_v, iota), base_v) };
    }

    KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_add_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_sub_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_mul_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_div_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) one_div(batch_t v) noexcept
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), v.v[0]) };
    }

    KSIMD_API(float64) reduce_add(batch_t v) noexcept
    {
        // [b, a]
        //   +
        // [a, b]
        //   =
        // [a+b]
        // get lane[0]

        __m128d sum64 = _mm_add_pd(v.v[0], _mm_shuffle_pd(v.v[0], v.v[0], _MM_SHUFFLE2(0, 1)));
        return _mm_cvtsd_f64(sum64);
    }

    KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
    {
        return { _mm_add_pd(_mm_mul_pd(a.v[0], b.v[0]), c.v[0]) };
    }

    KSIMD_API(batch_t) sqrt(batch_t v) noexcept
    {
        return { _mm_sqrt_pd(v.v[0]) };
    }

    KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
    {
        return { _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v.v[0])) };
    }

    KSIMD_API(batch_t) abs(batch_t v) noexcept
    {
        return { _mm_and_pd(v.v[0], _mm_set1_pd(SignBitClearMask<float64>)) };
    }

    KSIMD_API(batch_t) neg(batch_t v) noexcept
    {
        return { _mm_xor_pd(v.v[0], _mm_set1_pd(SignBitMask<float64>)) };
    }

    KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_min_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_max_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpeq_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpneq_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpgt_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpngt_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpge_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpnge_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmplt_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpnlt_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmple_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpnle_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpunord_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        __m128d l_nan = _mm_cmpunord_pd(lhs.v[0], lhs.v[0]);
        __m128d r_nan = _mm_cmpunord_pd(rhs.v[0], rhs.v[0]);
        return { _mm_and_pd(l_nan, r_nan) };
    }

    KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_cmpord_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m128d abs_mask = _mm_set1_pd(SignBitClearMask<float64>);
        __m128d inf = _mm_set1_pd(Inf<float64>);

        // 如果一个是有限值(指数位有0)，AND 之后结果的指数位一定会有0
        __m128d combined = _mm_and_pd(lhs.v[0], rhs.v[0]);

        return { _mm_cmplt_pd(_mm_and_pd(combined, abs_mask), inf) };
    }

    KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m128d abs_mask = _mm_set1_pd(SignBitClearMask<float64>);
        __m128d inf = _mm_set1_pd(Inf<float64>);

        __m128d l_finite = _mm_cmplt_pd(_mm_and_pd(lhs.v[0], abs_mask), inf);
        __m128d r_finite = _mm_cmplt_pd(_mm_and_pd(rhs.v[0], abs_mask), inf);

        return { _mm_and_pd(l_finite, r_finite) };
    }

    KSIMD_API(batch_t) bit_not(batch_t v) noexcept
    {
        return { _mm_xor_pd(v.v[0], _mm_set1_pd(OneBlock<float64>)) };
    }

    KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_and_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_andnot_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_or_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm_xor_pd(lhs.v[0], rhs.v[0]) };
    }

    KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm_or_pd(_mm_and_pd(mask.v[0], a.v[0]), _mm_andnot_pd(mask.v[0], b.v[0])) };
    }

    KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm_or_pd(_mm_and_pd(mask.m[0], a.v[0]), _mm_andnot_pd(mask.m[0], b.v[0])) };
    }
};
#undef KSIMD_API


#define KSIMD_API(ret) KSIMD_OP_SSE3_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float64>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float64>
{
    KSIMD_API(float64) reduce_add(batch_t v) noexcept
    {
        // input: [b, a]
        // hadd: [a+b]
        // get lane[0]
        __m128d result = _mm_hadd_pd(v.v[0], v.v[0]);
        return _mm_cvtsd_f64(result);
    }
};
#undef KSIMD_API


template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float64>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float64>
{};


#define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float64>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float64>
{
    KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm_blendv_pd(b.v[0], a.v[0], mask.m[0]) };
    }

    template<RoundingMode mode>
    KSIMD_API(batch_t) round(batch_t v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return { _mm_round_pd(v.v[0], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return { _mm_round_pd(v.v[0], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return { _mm_round_pd(v.v[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Round)
        {
            // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
            __m128d sign_mask = _mm_and_pd(v.v[0], _mm_set1_pd(SignBitMask<float32>));

            // 构造一个具有相同符号的0.5 (0x1.0p-1f == 0.5f 16进制精确表示)
            __m128d half = _mm_or_pd(_mm_set1_pd(0x1.0p-1f), sign_mask);

            return { _mm_round_pd(_mm_add_pd(v.v[0], half), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
        else /* if constexpr (mode == RoundingMode::ToZero) */
        {
            return { _mm_round_pd(v.v[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
    }
};
#undef KSIMD_API

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

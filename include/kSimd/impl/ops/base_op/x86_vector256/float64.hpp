#pragma once

#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 3.0, 2.0, 1.0, 0.0

KSIMD_NAMESPACE_BEGIN

// -------------------------------- operators --------------------------------
#define KSIMD_API(ret) KSIMD_OP_AVX_API ret KSIMD_CALL_CONV
namespace x86_vector256
{
    KSIMD_API(Batch<float64>) operator+(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator-(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator*(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator/(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator-(Batch<float64> v) noexcept
    {
        return { _mm256_xor_pd(v.v, _mm256_set1_pd(SignBitMask<float64>)) };
    }

    KSIMD_API(Batch<float64>) operator&(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator|(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator^(Batch<float64> lhs, Batch<float64> rhs) noexcept
    {
        return { _mm256_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float64>) operator~(Batch<float64> v) noexcept
    {
        return { _mm256_xor_pd(v.v, _mm256_set1_pd(OneBlock<float64>)) };
    }

    KSIMD_API(Batch<float64>&) operator+=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    KSIMD_API(Batch<float64>&) operator-=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    KSIMD_API(Batch<float64>&) operator*=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    KSIMD_API(Batch<float64>&) operator/=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    KSIMD_API(Batch<float64>&) operator&=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    KSIMD_API(Batch<float64>&) operator|=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    KSIMD_API(Batch<float64>&) operator^=(Batch<float64>& lhs, Batch<float64> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace x86_vector256
#undef KSIMD_API


#define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float64>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float64)

#if defined(KSIMD_IS_TESTING)
    KSIMD_API(void) test_store_mask(float64* mem, mask_t mask) noexcept
    {
        _mm256_store_pd(mem, mask.m);
    }
    KSIMD_API(mask_t) test_load_mask(const float64* mem) noexcept
    {
        return { _mm256_load_pd(mem) };
    }
#endif

    KSIMD_API(mask_t) mask_from_lanes(size_t count) noexcept
    {
        __m256d idx = _mm256_set_pd(KSIMD_IOTA);
        __m256d cnt = _mm256_set1_pd(static_cast<float64>(count));
        return { _mm256_cmp_pd(idx, cnt, _CMP_LT_OQ) };
    }

    KSIMD_API(batch_t) load(const float64* mem) noexcept
    {
        return { _mm256_load_pd(mem) };
    }

    KSIMD_API(batch_t) loadu(const float64* mem) noexcept
    {
        return { _mm256_loadu_pd(mem) };
    }

    KSIMD_API(void) store(float64* mem, batch_t v) noexcept
    {
        _mm256_store_pd(mem, v.v);
    }

    KSIMD_API(void) storeu(float64* mem, batch_t v) noexcept
    {
        _mm256_storeu_pd(mem, v.v);
    }

    KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask) noexcept
    {
        return { _mm256_maskload_pd(mem, _mm256_castpd_si256(mask.m)) };
    }

    KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask, batch_t default_value) noexcept
    {
        batch_t loaded = mask_load(mem, mask);
        return { _mm256_or_pd(loaded.v, _mm256_andnot_pd(mask.m, default_value.v)) };
    }

    KSIMD_API(batch_t) mask_loadu(const float64* mem, mask_t mask) noexcept
    {
        return { _mm256_maskload_pd(mem, _mm256_castpd_si256(mask.m)) };
    }

    KSIMD_API(batch_t) mask_loadu(const float64* mem, mask_t mask, batch_t default_value) noexcept
    {
        return mask_load(mem, mask, default_value);
    }

    KSIMD_API(void) mask_store(float64* mem, batch_t v, mask_t mask) noexcept
    {
        _mm256_maskstore_pd(mem, _mm256_castpd_si256(mask.m), v.v);
    }

    KSIMD_API(void) mask_storeu(float64* mem, batch_t v, mask_t mask) noexcept
    {
        _mm256_maskstore_pd(mem, _mm256_castpd_si256(mask.m), v.v);
    }

    KSIMD_API(batch_t) undefined() noexcept
    {
        return { _mm256_undefined_pd() };
    }

    KSIMD_API(batch_t) zero() noexcept
    {
        return { _mm256_setzero_pd() };
    }

    KSIMD_API(batch_t) set(float64 x) noexcept
    {
        return { _mm256_set1_pd(x) };
    }
    
    KSIMD_API(batch_t) sequence() noexcept
    {
        return { _mm256_set_pd(KSIMD_IOTA) };
    }

    KSIMD_API(batch_t) sequence(float64 base) noexcept
    {
        __m256d iota = _mm256_set_pd(KSIMD_IOTA);
        return { _mm256_add_pd(iota, _mm256_set1_pd(base)) };
    }

    KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
    {
        __m256d iota = _mm256_set_pd(KSIMD_IOTA);
        __m256d stride_v = _mm256_set1_pd(stride);
        __m256d base_v = _mm256_set1_pd(base);
        return { _mm256_add_pd(_mm256_mul_pd(stride_v, iota), base_v) };
    }

    KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_add_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_sub_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_mul_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_div_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) one_div(batch_t v) noexcept
    {
        return { _mm256_div_pd(_mm256_set1_pd(1.0), v.v) };
    }

    KSIMD_API(float64) reduce_add(batch_t v) noexcept
    {
        // [4,3] + [2,1] = [2+4, 1+3]
        __m128d low = _mm256_castpd256_pd128(v.v);
        __m128d high = _mm256_extractf128_pd(v.v, 0b1);
        __m128d sum128 = _mm_add_pd(low, high);

        // [2+4, 1+3] + [?, 2+4] = [1+2+3+4]
        __m128d sum64 = _mm_add_pd(sum128, _mm_unpackhi_pd(sum128, sum128));

        // get lane[0]
        return _mm_cvtsd_f64(sum64);
    }

    KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
    {
        return { _mm256_add_pd(_mm256_mul_pd(a.v, b.v), c.v) };
    }

    KSIMD_API(batch_t) sqrt(batch_t v) noexcept
    {
        return { _mm256_sqrt_pd(v.v) };
    }

    KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
    {
        return { _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(v.v)) };
    }

    template<RoundingMode mode>
    KSIMD_API(batch_t) round(batch_t v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return { _mm256_round_pd(v.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return { _mm256_round_pd(v.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return { _mm256_round_pd(v.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
        }
        else if constexpr (mode == RoundingMode::Round)
        {
            // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
            __m256d sign_mask = _mm256_and_pd(v.v, _mm256_set1_pd(SignBitMask<float32>));

            // 构造一个具有相同符号的0.5
            __m256d half = _mm256_or_pd(_mm256_set1_pd(0x1.0p-1f), sign_mask);

            return { _mm256_round_pd(_mm256_add_pd(v.v, half), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
        else /* if constexpr (mode == RoundingMode::ToZero) */
        {
            return { _mm256_round_pd(v.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
    }

    KSIMD_API(batch_t) abs(batch_t v) noexcept
    {
        return { _mm256_and_pd(v.v, _mm256_set1_pd(SignBitClearMask<float64>)) };
    }

    KSIMD_API(batch_t) neg(batch_t v) noexcept
    {
        return { _mm256_xor_pd(v.v, _mm256_set1_pd(SignBitMask<float64>)) };
    }

    KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_min_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_max_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        __m256d l_nan = _mm256_cmp_pd(lhs.v, lhs.v, _CMP_UNORD_Q);
        __m256d r_nan = _mm256_cmp_pd(rhs.v, rhs.v, _CMP_UNORD_Q);
        return { _mm256_and_pd(l_nan, r_nan) };
    }

    KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_pd(lhs.v, rhs.v, _CMP_ORD_Q) };
    }

    KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
        __m256d inf = _mm256_set1_pd(Inf<float64>);

        __m256d combined = _mm256_and_pd(lhs.v, rhs.v);

        return { _mm256_cmp_pd(_mm256_and_pd(combined, abs_mask), inf, _CMP_LT_OQ) };
    }

    KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
        __m256d inf = _mm256_set1_pd(Inf<float64>);

        __m256d l_finite = _mm256_cmp_pd(_mm256_and_pd(lhs.v, abs_mask), inf, _CMP_LT_OQ);
        __m256d r_finite = _mm256_cmp_pd(_mm256_and_pd(rhs.v, abs_mask), inf, _CMP_LT_OQ);

        return { _mm256_and_pd(l_finite, r_finite) };
    }

    KSIMD_API(batch_t) bit_not(batch_t v) noexcept
    {
        return { _mm256_xor_pd(v.v, _mm256_set1_pd(OneBlock<float64>)) };
    }

    KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_and_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_andnot_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_or_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_xor_pd(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm256_or_pd(_mm256_and_pd(mask.v, a.v), _mm256_andnot_pd(mask.v, b.v)) };
    }

    KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm256_blendv_pd(b.v, a.v, mask.m) };
    }
};
#undef KSIMD_API


template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float64>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float64>
{};


#define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float64>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float64>
{
private:
    using base = BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float64>;

public:
    using base::sequence;

    KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
    {
        __m256d iota = _mm256_set_pd(KSIMD_IOTA);
        __m256d stride_v = _mm256_set1_pd(stride);
        __m256d base_v = _mm256_set1_pd(base);
        return { _mm256_fmadd_pd(stride_v, iota, base_v) };
    }

    KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
    {
        return { _mm256_fmadd_pd(a.v, b.v, c.v) };
    }
};
#undef KSIMD_API

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

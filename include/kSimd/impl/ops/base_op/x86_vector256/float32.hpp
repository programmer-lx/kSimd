#pragma once

#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/utils.hpp"

KSIMD_NAMESPACE_BEGIN

// -------------------------------- operators --------------------------------
#define KSIMD_API(ret) KSIMD_OP_AVX_API ret KSIMD_CALL_CONV
namespace x86_vector256
{
    KSIMD_API(Batch<float32>) operator+(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator-(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator*(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator/(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator-(Batch<float32> v) noexcept
    {
        return { _mm256_sub_ps(_mm256_setzero_ps(), v.v) };
    }

    KSIMD_API(Batch<float32>) operator&(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator|(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator^(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) operator~(Batch<float32> v) noexcept
    {
        return { _mm256_xor_ps(v.v, _mm256_set1_ps(OneBlock<float32>)) };
    }

    KSIMD_API(Batch<float32>&) operator+=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    KSIMD_API(Batch<float32>&) operator-=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    KSIMD_API(Batch<float32>&) operator*=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    KSIMD_API(Batch<float32>&) operator/=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    KSIMD_API(Batch<float32>&) operator&=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    KSIMD_API(Batch<float32>&) operator|=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    KSIMD_API(Batch<float32>&) operator^=(Batch<float32>& lhs, Batch<float32> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace x86_vector256
#undef KSIMD_API


#define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::AVX, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::AVX, float32)

#if defined(KSIMD_IS_TESTING)
    KSIMD_API(void) test_store_mask(float32* mem, mask_t mask) noexcept
    {
        _mm256_store_ps(mem, mask.m);
    }
    KSIMD_API(mask_t) test_load_mask(const float32* mem) noexcept
    {
        return { _mm256_load_ps(mem) };
    }
#endif

    KSIMD_API(mask_t) mask_from_lanes(unsigned int count) noexcept
    {
        __m256 idx = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
        __m256 cnt = _mm256_set1_ps(static_cast<float32>(count));
        return { _mm256_cmp_ps(idx, cnt, _CMP_LT_OQ) };
    }

    KSIMD_API(batch_t) load(const float32* mem) noexcept
    {
        return { _mm256_load_ps(mem) };
    }

    KSIMD_API(batch_t) loadu(const float32* mem) noexcept
    {
        return { _mm256_loadu_ps(mem) };
    }

    KSIMD_API(void) store(float32* mem, batch_t v) noexcept
    {
        _mm256_store_ps(mem, v.v);
    }

    KSIMD_API(void) storeu(float32* mem, batch_t v) noexcept
    {
        _mm256_storeu_ps(mem, v.v);
    }

    KSIMD_API(batch_t) mask_load(const float32* mem, mask_t mask) noexcept
    {
        return { _mm256_maskload_ps(mem, _mm256_castps_si256(mask.m)) };
    }

    KSIMD_API(batch_t) mask_loadu(const float32* mem, mask_t mask) noexcept
    {
        return { _mm256_maskload_ps(mem, _mm256_castps_si256(mask.m)) };
    }

    KSIMD_API(void) mask_store(float32* mem, batch_t v, mask_t mask) noexcept
    {
        _mm256_maskstore_ps(mem, _mm256_castps_si256(mask.m), v.v);
    }

    KSIMD_API(void) mask_storeu(float32* mem, batch_t v, mask_t mask) noexcept
    {
        _mm256_maskstore_ps(mem, _mm256_castps_si256(mask.m), v.v);
    }

    KSIMD_API(batch_t) undefined() noexcept
    {
        return { _mm256_undefined_ps() };
    }

    KSIMD_API(batch_t) zero() noexcept
    {
        return { _mm256_setzero_ps() };
    }

    KSIMD_API(batch_t) set(float32 x) noexcept
    {
        return { _mm256_set1_ps(x) };
    }

    KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) one_div(batch_t v) noexcept
    {
        return { _mm256_rcp_ps(v.v) };
    }

    KSIMD_API(float32) reduce_add(batch_t v) noexcept
    {
        // [8,7,6,5] + [4,3,2,1] = [8+4, 7+3, 6+2, 5+1]
        __m128 low = _mm256_castps256_ps128(v.v);
        __m128 high = _mm256_extractf128_ps(v.v, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        // [8+4, 7+3, 8+4, 7+3] + [8+4, 7+3, 6+2, 5+1] = [?, ?, 2468, 1357]
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));

        // [?, ?, 2468, 1357] + [?, ?, 2468, 2468] = [?, ?, ?, 12345678]
        __m128 res = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 1, 1, 1)));

        // get lane[0]
        return _mm_cvtss_f32(res);
    }

    KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
    {
        return { _mm256_add_ps(_mm256_mul_ps(a.v, b.v), c.v) };
    }

    KSIMD_API(batch_t) sqrt(batch_t v) noexcept
    {
        return { _mm256_sqrt_ps(v.v) };
    }

    KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
    {
        return { _mm256_rsqrt_ps(v.v) };
    }

    KSIMD_API(batch_t) round_up(batch_t v) noexcept
    {
        return { _mm256_round_ps(v.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) };
    }

    KSIMD_API(batch_t) round_down(batch_t v) noexcept
    {
        return { _mm256_round_ps(v.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) };
    }

    KSIMD_API(batch_t) round_nearest(batch_t v) noexcept
    {
        return { _mm256_round_ps(v.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
    }

    KSIMD_API(batch_t) round(batch_t v) noexcept
    {
        // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
        __m256 sign_mask = _mm256_and_ps(v.v, _mm256_set1_ps(SignBitMask<float32>));

        // 构造一个具有相同符号的0.5，但是比0.5小一点，防止进位
        __m256 half = _mm256_or_ps(_mm256_set1_ps(0.49999997f), sign_mask);

        return { _mm256_round_ps(_mm256_add_ps(v.v, half), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
    }

    KSIMD_API(batch_t) round_to_zero(batch_t v) noexcept
    {
        return { _mm256_round_ps(v.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
    }

    KSIMD_API(batch_t) abs(batch_t v) noexcept
    {
        // 将 sign bit 反转即可
        return { _mm256_and_ps(v.v, _mm256_set1_ps(SignBitClearMask<float32>)) };
    }

    KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_min_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_max_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        __m256 l_nan = _mm256_cmp_ps(lhs.v, lhs.v, _CMP_UNORD_Q);
        __m256 r_nan = _mm256_cmp_ps(rhs.v, rhs.v, _CMP_UNORD_Q);
        return { _mm256_and_ps(l_nan, r_nan) };
    }

    KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_ORD_Q) };
    }

    KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
        __m256 inf = _mm256_set1_ps(Inf<float32>);

        __m256 combined = _mm256_and_ps(lhs.v, rhs.v);

        return { _mm256_cmp_ps(_mm256_and_ps(combined, abs_mask), inf, _CMP_LT_OQ) };
    }

    KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
        __m256 inf = _mm256_set1_ps(Inf<float32>);

        __m256 l_finite = _mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf, _CMP_LT_OQ);
        __m256 r_finite = _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf, _CMP_LT_OQ);

        return { _mm256_and_ps(l_finite, r_finite) };
    }

    KSIMD_API(batch_t) bit_not(batch_t v) noexcept
    {
        return { _mm256_xor_ps(v.v, _mm256_set1_ps(OneBlock<float32>)) };
    }

    KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_andnot_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm256_or_ps(_mm256_and_ps(mask.v, a.v), _mm256_andnot_ps(mask.v, b.v)) };
    }

    KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
    {
        return { _mm256_blendv_ps(b.v, a.v, mask.m) };
    }
};
#undef KSIMD_API


template<>
struct BaseOp<SimdInstruction::AVX2, float32> : BaseOp<SimdInstruction::AVX, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::AVX2, float32)
};


// AVX2 + FMA指令特化
#define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_F16C_API static ret KSIMD_CALL_CONV
template<>
struct BaseOp<SimdInstruction::AVX2_FMA3_F16C, float32> : BaseOp<SimdInstruction::AVX2, float32>
{
    KSIMD_DETAIL_BASE_OP_TRAITS(SimdInstruction::AVX2_FMA3_F16C, float32)

    KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
    {
        return { _mm256_fmadd_ps(a.v, b.v, c.v) };
    }
};
#undef KSIMD_API

KSIMD_NAMESPACE_END

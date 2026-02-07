#pragma once

#include "traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_API(...) KSIMD_OP_AVX2_FMA3_F16C_API static __VA_ARGS__ KSIMD_CALL_CONV

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // AVX2_FMA3_F16C
    template<typename Traits, typename = void>
    struct Executor_AVX2_FMA3_F16C_Impl_float32;
    
    template<typename Traits, size_t... I>
    struct Executor_AVX2_FMA3_F16C_Impl_float32<Traits, std::index_sequence<I...>> : BaseOpHelper
    {
        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float32* mem, typename Traits::mask_t mask) noexcept
        {
            (_mm256_store_ps(&mem[I * Traits::RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(typename Traits::mask_t) test_load_mask(const float32* mem) noexcept
        {
            return { _mm256_load_ps(&mem[I * Traits::RegLanes])... };
        }
        #endif

        KSIMD_API(typename Traits::batch_t) load(const float32* mem) noexcept
        {
            return { _mm256_load_ps(&mem[I * Traits::RegLanes])... };
        }

        KSIMD_API(typename Traits::batch_t) loadu(const float32* mem) noexcept
        {
            return { _mm256_loadu_ps(&mem[I * Traits::RegLanes])... };
        }

        KSIMD_API(typename Traits::batch_t) load_partial(const float32* mem, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;

            if (count == 0) [[unlikely]]
                return zero();

            typename Traits::batch_t res = zero();
            std::memcpy(res.v, mem, sizeof(float32) * count);
            return res;
        }

        KSIMD_API(void) store(float32* mem, typename Traits::batch_t v) noexcept
        {
            (_mm256_store_ps(&mem[I * Traits::RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float32* mem, typename Traits::batch_t v) noexcept
        {
            (_mm256_storeu_ps(&mem[I * Traits::RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) store_partial(float32* mem, typename Traits::batch_t v, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;
            if (count == 0) [[unlikely]]
                return;

            std::memcpy(mem, v.v, sizeof(float32) * count);
        }

        KSIMD_API(typename Traits::batch_t) undefined() noexcept
        {
            return { ((void)I, _mm256_undefined_ps())... };
        }

        KSIMD_API(typename Traits::batch_t) zero() noexcept
        {
            return { ((void)I, _mm256_setzero_ps())... };
        }

        KSIMD_API(typename Traits::batch_t) set(float32 x) noexcept
        {
            return { ((void)I, _mm256_set1_ps(x))... };
        }

        KSIMD_API(typename Traits::batch_t) add(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_add_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) sub(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_sub_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) mul(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_mul_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) div(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_div_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) one_div(typename Traits::batch_t v) noexcept
        {
            return { _mm256_rcp_ps(v.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) mul_add(typename Traits::batch_t a, typename Traits::batch_t b, typename Traits::batch_t c) noexcept
        {
            return { _mm256_fmadd_ps(a.v[I], b.v[I], c.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) sqrt(typename Traits::batch_t v) noexcept
        {
            return { _mm256_sqrt_ps(v.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) rsqrt(typename Traits::batch_t v) noexcept
        {
            return { _mm256_rsqrt_ps(v.v[I])... };
        }

        template<RoundingMode mode>
        KSIMD_API(typename Traits::batch_t) round(typename Traits::batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return { _mm256_round_ps(v.v[I], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return { _mm256_round_ps(v.v[I], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return { _mm256_round_ps(v.v[I], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
                __m256 sign_bit = _mm256_set1_ps(SignBitMask<float32>);
                // __m256 sign_mask = _mm256_and_ps(v.v[I], sign_bit);

                // 构造一个具有相同符号的0.5
                // __m256 half = _mm256_or_ps(_mm256_set1_ps(0x1.0p-1f), _mm256_and_ps(v.v[I], sign_bit));

                return { _mm256_round_ps(_mm256_add_ps(v.v[I],
                    _mm256_or_ps(_mm256_set1_ps(0x1.0p-1f), _mm256_and_ps(v.v[I], sign_bit))),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return { _mm256_round_ps(v.v[I], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
        }

        KSIMD_API(typename Traits::batch_t) abs(typename Traits::batch_t v) noexcept
        {
            // 将 sign bit 反转即可
            return { _mm256_and_ps(v.v[I], _mm256_set1_ps(SignBitClearMask<float32>))... };
        }

        KSIMD_API(typename Traits::batch_t) neg(typename Traits::batch_t v) noexcept
        {
            __m256 mask = _mm256_set1_ps(SignBitMask<float32>);
            return { _mm256_xor_ps(v.v[I], mask)... };
        }

        KSIMD_API(typename Traits::batch_t) min(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_min_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) max(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_max_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::mask_t) equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_EQ_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NEQ_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_GT_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NGT_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_GE_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NGE_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_LT_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NLT_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_LE_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NLE_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) any_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_UNORD_Q)... };
        }

        KSIMD_API(typename Traits::mask_t) all_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            // __m256 l_nan = _mm256_cmp_ps(lhs.v[I], lhs.v[I], _CMP_UNORD_Q);
            // __m256 r_nan = _mm256_cmp_ps(rhs.v[I], rhs.v[I], _CMP_UNORD_Q);
            return { _mm256_and_ps(
                _mm256_cmp_ps(lhs.v[I], lhs.v[I], _CMP_UNORD_Q),
                _mm256_cmp_ps(rhs.v[I], rhs.v[I], _CMP_UNORD_Q))... };
        }

        KSIMD_API(typename Traits::mask_t) not_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_ORD_Q)... };
        }

        KSIMD_API(typename Traits::mask_t) any_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
            __m256 inf = _mm256_set1_ps(Inf<float32>);
            return {
                _mm256_or_ps(
                    _mm256_cmp_ps(_mm256_and_ps(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                    _mm256_cmp_ps(_mm256_and_ps(rhs.v[I], abs_mask), inf, _CMP_LT_OQ)
                )...
            };
        }

        KSIMD_API(typename Traits::mask_t) all_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
            __m256 inf = _mm256_set1_ps(Inf<float32>);

            // __m256 l_finite = _mm256_cmp_ps(_mm256_and_ps(lhs.v[I], abs_mask), inf, _CMP_LT_OQ);
            // __m256 r_finite = _mm256_cmp_ps(_mm256_and_ps(rhs.v[I], abs_mask), inf, _CMP_LT_OQ);

            return { _mm256_and_ps(
                _mm256_cmp_ps(_mm256_and_ps(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                _mm256_cmp_ps(_mm256_and_ps(rhs.v[I], abs_mask), inf, _CMP_LT_OQ))... };
        }

        KSIMD_API(typename Traits::batch_t) bit_not(typename Traits::batch_t v) noexcept
        {
            __m256 mask = _mm256_set1_ps(OneBlock<float32>);
            return { _mm256_xor_ps(v.v[I], mask)... };
        }

        KSIMD_API(typename Traits::batch_t) bit_and(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_and_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_and_not(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_andnot_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_or(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_or_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_xor(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_xor_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_select(typename Traits::batch_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            return { _mm256_or_ps(_mm256_and_ps(mask.v[I], a.v[I]), _mm256_andnot_ps(mask.v[I], b.v[I]))... };
        }

        KSIMD_API(typename Traits::batch_t) mask_select(typename Traits::mask_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            return { _mm256_blendv_ps(b.v[I], a.v[I], mask.m[I])... };
        }
    };

    template<typename Traits, size_t reg_count>
    using Executor_AVX2_FMA3_F16C_float32 = Executor_AVX2_FMA3_F16C_Impl_float32<Traits, std::make_index_sequence<reg_count>>;
}

// -------------------------------- operators --------------------------------
#define KSIMD_EXE detail::Executor_AVX2_FMA3_F16C_float32<BaseOpTraits_AVX_Family<float32, 1, x86_vector256::Mask<float32, 1>>, 1>
namespace x86_vector256
{
    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator+(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::add(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::sub(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator*(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::mul(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator/(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::div(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> v) noexcept
    {
        return KSIMD_EXE::neg(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator&(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::bit_and(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator|(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::bit_or(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator^(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return KSIMD_EXE::bit_xor(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator~(Batch<float32, reg_count> v) noexcept
    {
        return KSIMD_EXE::bit_not(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator+=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator-=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator*=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator/=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator&=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator|=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator^=(Batch<float32, reg_count>& lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace x86_vector256
#undef KSIMD_EXE

// base op mixin
#define KSIMD_BATCH_T x86_vector256::Batch<float32, 1>
namespace detail
{
    struct Base_Mixin_AVX2_FMA3_F16C_float32
    {
        KSIMD_API(float32) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            __m128 low = _mm256_castps256_ps128(v.v[0]); // [1, 2, 3, 4]
            __m128 high = _mm256_extractf128_ps(v.v[0], 0b1); // [5, 6, 7, 8]
            __m128 sum = _mm_add_ps(low, high); // [1+5, 2+6, 3+7, 4+8]

            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }

        KSIMD_API(KSIMD_BATCH_T) sequence() noexcept
        {
            return { _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base) noexcept
        {
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
            return { _mm256_add_ps(iota, base_v) };
        }

        KSIMD_API(KSIMD_BATCH_T) sequence(float32 base, float32 stride) noexcept
        {
            __m256 stride_v = _mm256_set1_ps(stride);
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
            return { _mm256_fmadd_ps(stride_v, iota, base_v) };
        }
    };
}
#undef KSIMD_BATCH_T

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C, float32>
    : BaseOpTraits_AVX_Family<float32, 1, x86_vector256::Mask<float32, 1>>
    , detail::Executor_AVX2_FMA3_F16C_float32<BaseOpTraits_AVX_Family<float32, 1, x86_vector256::Mask<float32, 1>>, 1>
    , detail::Base_Mixin_AVX2_FMA3_F16C_float32
{};

KSIMD_NAMESPACE_END

#undef KSIMD_API
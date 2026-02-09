#pragma once

#include "traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/op/Op.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_API(...) KSIMD_OP_AVX2_FMA3_F16C_API static __VA_ARGS__ KSIMD_CALL_CONV

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // AVX2_FMA3_F16C
    template<typename Traits, typename = void>
    struct Executor_AVX2_FMA3_F16C_float64_Impl;

    template<typename Traits, size_t... I>
    struct Executor_AVX2_FMA3_F16C_float64_Impl<Traits, std::index_sequence<I...>> : OpHelper
    {
        static_assert(std::is_same_v<typename Traits::scalar_t, float64>);

        KSIMD_API(typename Traits::batch_t) load(const float64* mem) noexcept
        {
            return { _mm256_load_pd(&mem[I * Traits::RegLanes])... };
        }

        KSIMD_API(typename Traits::batch_t) loadu(const float64* mem) noexcept
        {
            return { _mm256_loadu_pd(&mem[I * Traits::RegLanes])... };
        }

        KSIMD_API(typename Traits::batch_t) load_partial(const float64* mem, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;

            if (count == 0) [[unlikely]]
                return zero();

            typename Traits::batch_t res = zero();
            std::memcpy(res.v, mem, sizeof(float64) * count);
            return res;
        }

        KSIMD_API(void) store(float64* mem, typename Traits::batch_t v) noexcept
        {
            (_mm256_store_pd(&mem[I * Traits::RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float64* mem, typename Traits::batch_t v) noexcept
        {
            (_mm256_storeu_pd(&mem[I * Traits::RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) store_partial(float64* mem, typename Traits::batch_t v, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;
            if (count == 0) [[unlikely]]
                return;

            std::memcpy(mem, v.v, sizeof(float64) * count);
        }

        KSIMD_API(typename Traits::batch_t) undefined() noexcept
        {
            __m256d u = _mm256_undefined_pd();
            return { ((void)I, u)... };
        }

        KSIMD_API(typename Traits::batch_t) zero() noexcept
        {
            __m256d z = _mm256_setzero_pd();
            return { ((void)I, z)... };
        }

        KSIMD_API(typename Traits::batch_t) set(float64 x) noexcept
        {
            __m256d v = _mm256_set1_pd(x);
            return { ((void)I, v)... };
        }

        KSIMD_API(typename Traits::batch_t) add(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_add_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) sub(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_sub_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) mul(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_mul_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) div(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_div_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) one_div(typename Traits::batch_t v) noexcept
        {
            return { _mm256_div_pd(_mm256_set1_pd(1.0), v.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) mul_add(typename Traits::batch_t a, typename Traits::batch_t b, typename Traits::batch_t c) noexcept
        {
            return { _mm256_fmadd_pd(a.v[I], b.v[I], c.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) sqrt(typename Traits::batch_t v) noexcept
        {
            return { _mm256_sqrt_pd(v.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) rsqrt(typename Traits::batch_t v) noexcept
        {
            __m256d one = _mm256_set1_pd(1.0);
            return { _mm256_div_pd(one, _mm256_sqrt_pd(v.v[I]))... };
        }

        template<RoundingMode mode>
        KSIMD_API(typename Traits::batch_t) round(typename Traits::batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return { _mm256_round_pd(v.v[I], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return { _mm256_round_pd(v.v[I], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return { _mm256_round_pd(v.v[I], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
                __m256d sign_bit = _mm256_set1_pd(SignBitMask<float64>);
                // __m256d sign_mask = _mm256_and_pd(v.v[I], sign_bit);

                // 0.5
                __m256d half = _mm256_set1_pd(0x1.0p-1);

                return { _mm256_round_pd(_mm256_add_pd(v.v[I],
                    _mm256_or_pd(half, _mm256_and_pd(v.v[I], sign_bit))),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return { _mm256_round_pd(v.v[I], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
        }

        KSIMD_API(typename Traits::batch_t) abs(typename Traits::batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(SignBitClearMask<float64>);
            return { _mm256_and_pd(v.v[I], mask)... };
        }

        KSIMD_API(typename Traits::batch_t) neg(typename Traits::batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(SignBitMask<float64>);
            return { _mm256_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(typename Traits::batch_t) min(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_min_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) max(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_max_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_not(typename Traits::batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(OneBlock<float64>);
            return { _mm256_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(typename Traits::batch_t) bit_and(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_and_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_and_not(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_andnot_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_or(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_or_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_xor(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_xor_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(typename Traits::batch_t) bit_select(typename Traits::batch_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            return { _mm256_or_pd(_mm256_and_pd(mask.v[I], a.v[I]), _mm256_andnot_pd(mask.v[I], b.v[I]))... };
        }
    };

    template<typename Traits, size_t reg_count>
    using Executor_AVX2_FMA3_F16C_float64 = Executor_AVX2_FMA3_F16C_float64_Impl<Traits, std::make_index_sequence<reg_count>>;
}

// -------------------------------- operators --------------------------------
namespace x86_vector256
{
    #define KSIMD_UNROLL_BINARY_OP(intrinsic) \
        static_assert(reg_count <= 2, "512bit is max."); \
        if constexpr (reg_count == 1) { return { intrinsic(lhs.v[0], rhs.v[0]) }; } \
        else { return { intrinsic(lhs.v[0], rhs.v[0]), intrinsic(lhs.v[1], rhs.v[1]) }; }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator+(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_add_pd)
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_sub_pd)
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator*(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_mul_pd)
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator/(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_div_pd)
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator&(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_and_pd)
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator|(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_or_pd)
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator^(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        KSIMD_UNROLL_BINARY_OP(_mm256_xor_pd)
    }

    #undef KSIMD_UNROLL_BINARY_OP

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> v) noexcept
    {
        static_assert(reg_count <= 2, "512bit is max.");

        __m256d mask = _mm256_set1_pd(SignBitMask<float64>);
        if constexpr (reg_count == 1)
        {
            return { _mm256_xor_pd(v.v[0], mask) };
        }
        else
        {
            return { _mm256_xor_pd(v.v[0], mask), _mm256_xor_pd(v.v[1], mask) };
        }
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator~(Batch<float64, reg_count> v) noexcept
    {
        static_assert(reg_count <= 2, "512bit is max.");

        __m256d mask = _mm256_set1_pd(OneBlock<float64>);
        if constexpr (reg_count == 1)
        {
            return { _mm256_xor_pd(v.v[0], mask) };
        }
        else
        {
            return { _mm256_xor_pd(v.v[0], mask), _mm256_xor_pd(v.v[1], mask) };
        }
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
} // namespace x86_vector256

// base op horizontal operations mixin
namespace detail
{
    template<typename Traits>
    struct Base_Mixin_AVX2_FMA3_float64
    {
        static_assert(std::is_same_v<typename Traits::scalar_t, float64>);

        KSIMD_API(float64) reduce_add(typename Traits::batch_t v) noexcept
        {
            __m128d low = _mm256_castpd256_pd128(v.v[0]);
            __m128d high = _mm256_extractf128_pd(v.v[0], 0b1);
            __m128d sum128 = _mm_add_pd(low, high);

            // [2+4, 1+3] + [?, 2+4] = [1+2+3+4]
            __m128d sum64 = _mm_add_pd(sum128, _mm_unpackhi_pd(sum128, sum128));

            return _mm_cvtsd_f64(sum64);
        }

        KSIMD_API(typename Traits::batch_t) sequence() noexcept
        {
            return { _mm256_set_pd(3.0, 2.0, 1.0, 0.0) };
        }

        KSIMD_API(typename Traits::batch_t) sequence(float64 base) noexcept
        {
            __m256d base_v = _mm256_set1_pd(base);
            __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
            return { _mm256_add_pd(iota, base_v) };
        }

        KSIMD_API(typename Traits::batch_t) sequence(float64 base, float64 stride) noexcept
        {
            __m256d stride_v = _mm256_set1_pd(stride);
            __m256d base_v = _mm256_set1_pd(base);
            __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
            return { _mm256_fmadd_pd(stride_v, iota, base_v) };
        }
    };
}

// mask operations mixin
namespace detail
{
    template<typename Traits, typename = void>
    struct Base_Mixin_Mask_m256d_AVX2_FMA3_F16C_float64_Impl;

    template<typename Traits, size_t... I>
    struct Base_Mixin_Mask_m256d_AVX2_FMA3_F16C_float64_Impl<Traits, std::index_sequence<I...>>
    {
        static_assert(std::is_same_v<typename Traits::scalar_t, float64>);

        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float64* mem, typename Traits::mask_t mask) noexcept
        {
            (_mm256_store_pd(&mem[I * Traits::RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(typename Traits::mask_t) test_load_mask(const float64* mem) noexcept
        {
            return { _mm256_load_pd(&mem[I * Traits::RegLanes])... };
        }
        #endif

        KSIMD_API(typename Traits::mask_t) equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_EQ_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NEQ_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_GT_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NGT_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_GE_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NGE_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_LT_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NLT_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_LE_OQ)... };
        }

        KSIMD_API(typename Traits::mask_t) not_less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NLE_UQ)... };
        }

        KSIMD_API(typename Traits::mask_t) any_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_UNORD_Q)... };
        }

        KSIMD_API(typename Traits::mask_t) all_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            // __m256d l_nan = _mm256_cmp_pd(lhs.v[I], lhs.v[I], _CMP_UNORD_Q);
            // __m256d r_nan = _mm256_cmp_pd(rhs.v[I], rhs.v[I], _CMP_UNORD_Q);
            return { _mm256_and_pd(
                _mm256_cmp_pd(lhs.v[I], lhs.v[I], _CMP_UNORD_Q),
                _mm256_cmp_pd(rhs.v[I], rhs.v[I], _CMP_UNORD_Q))... };
        }

        KSIMD_API(typename Traits::mask_t) not_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_ORD_Q)... };
        }

        KSIMD_API(typename Traits::mask_t) any_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
            __m256d inf = _mm256_set1_pd(Inf<float64>);
            return {
                _mm256_or_pd(
                    _mm256_cmp_pd(_mm256_and_pd(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                    _mm256_cmp_pd(_mm256_and_pd(rhs.v[I], abs_mask), inf, _CMP_LT_OQ)
                )...
            };
        }

        KSIMD_API(typename Traits::mask_t) all_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
            __m256d inf = _mm256_set1_pd(Inf<float64>);

            // __m256d l_finite = _mm256_cmp_pd(_mm256_and_pd(lhs.v[I], abs_mask), inf, _CMP_LT_OQ);
            // __m256d r_finite = _mm256_cmp_pd(_mm256_and_pd(rhs.v[I], abs_mask), inf, _CMP_LT_OQ);

            return { _mm256_and_pd(
                _mm256_cmp_pd(_mm256_and_pd(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                _mm256_cmp_pd(_mm256_and_pd(rhs.v[I], abs_mask), inf, _CMP_LT_OQ))... };
        }

        KSIMD_API(typename Traits::batch_t) mask_select(typename Traits::mask_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            return { _mm256_blendv_pd(b.v[I], a.v[I], mask.m[I])... };
        }
    };

    template<typename Traits, size_t reg_count>
    using Base_Mixin_Mask_m256d_AVX2_FMA3_F16C_float64 = Base_Mixin_Mask_m256d_AVX2_FMA3_F16C_float64_Impl<Traits, std::make_index_sequence<reg_count>>;
}

#define KSIMD_TRAITS BaseOpTraits_AVX_Family<float64, 1, x86_vector256::Mask<float64, 1>>
template<>
struct Op<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C, float64>
    // traits
    : KSIMD_TRAITS

    // executor
    , detail::Executor_AVX2_FMA3_F16C_float64<KSIMD_TRAITS, 1>

    // __m256d mask mixin
    , detail::Base_Mixin_Mask_m256d_AVX2_FMA3_F16C_float64<KSIMD_TRAITS, 1>

    // horizontal mixin
    , detail::Base_Mixin_AVX2_FMA3_float64<KSIMD_TRAITS>
{};
#undef KSIMD_TRAITS

KSIMD_NAMESPACE_END

#undef KSIMD_API
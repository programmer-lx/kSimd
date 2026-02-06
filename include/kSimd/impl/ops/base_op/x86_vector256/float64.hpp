#pragma once

#include "traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 3.0, 2.0, 1.0, 0.0

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // AVX
    template<typename = void>
    struct Executor_AVX_Impl_float64;

    #define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_AVX_Impl_float64<std::index_sequence<I...>>
        : BaseOpHelper
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_AVX_Family<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float64, sizeof...(I)>)

        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float64* mem, mask_t mask) noexcept
        {
            (_mm256_store_pd(&mem[I * RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(mask_t) test_load_mask(const float64* mem) noexcept
        {
            return { _mm256_load_pd(&mem[I * RegLanes])... };
        }
        #endif

        KSIMD_API(mask_t) mask_from_lanes(size_t count) noexcept
        {
            __m256d cnt = _mm256_set1_pd(static_cast<float64>(count));
            __m256d idx = _mm256_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm256_cmp_pd(idx, cnt, _CMP_LT_OQ))... };
        }

        KSIMD_API(batch_t) load(const float64* mem) noexcept
        {
            return { _mm256_load_pd(&mem[I * RegLanes])... };
        }

        KSIMD_API(batch_t) loadu(const float64* mem) noexcept
        {
            return { _mm256_loadu_pd(&mem[I * RegLanes])... };
        }

        KSIMD_API(void) store(float64* mem, batch_t v) noexcept
        {
            (_mm256_store_pd(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float64* mem, batch_t v) noexcept
        {
            (_mm256_storeu_pd(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask) noexcept
        {
            return { _mm256_maskload_pd(&mem[I * RegLanes], _mm256_castpd_si256(mask.m[I]))... };
        }

        KSIMD_API(batch_t) mask_load(const float64* mem, mask_t mask, batch_t default_value) noexcept
        {
            batch_t loaded = mask_load(mem, mask);
            return { _mm256_or_pd(loaded.v[I], _mm256_andnot_pd(mask.m[I], default_value.v[I]))... };
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
            (_mm256_maskstore_pd(&mem[I * RegLanes], _mm256_castpd_si256(mask.m[I]), v.v[I]), ...);
        }

        KSIMD_API(void) mask_storeu(float64* mem, batch_t v, mask_t mask) noexcept
        {
            mask_store(mem, v, mask);
        }

        KSIMD_API(batch_t) undefined() noexcept
        {
            return { ((void)I, _mm256_undefined_pd())... };
        }

        KSIMD_API(batch_t) zero() noexcept
        {
            return { ((void)I, _mm256_setzero_pd())... };
        }

        KSIMD_API(batch_t) set(float64 x) noexcept
        {
            return { ((void)I, _mm256_set1_pd(x))... };
        }

        KSIMD_API(batch_t) sequence() noexcept
        {
            __m256d iota = _mm256_set_pd(KSIMD_IOTA);
            return { ((void)I, iota)... };
        }

        KSIMD_API(batch_t) sequence(float64 base) noexcept
        {
            __m256d base_v = _mm256_set1_pd(base);
            __m256d iota = _mm256_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm256_add_pd(iota, base_v))... };
        }

        KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
        {
            __m256d stride_v = _mm256_set1_pd(stride);
            __m256d base_v = _mm256_set1_pd(base);
            __m256d iota = _mm256_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm256_add_pd(_mm256_mul_pd(stride_v, iota), base_v))... };
        }

        KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_add_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_sub_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_mul_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_div_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) one_div(batch_t v) noexcept
        {
            return { _mm256_div_pd(_mm256_set1_pd(1.0), v.v[I])... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm256_add_pd(_mm256_mul_pd(a.v[I], b.v[I]), c.v[I])... };
        }

        KSIMD_API(batch_t) sqrt(batch_t v) noexcept
        {
            return { _mm256_sqrt_pd(v.v[I])... };
        }

        KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
        {
            __m256d one = _mm256_set1_pd(1.0);
            return { _mm256_div_pd(one, _mm256_sqrt_pd(v.v[I]))... };
        }

        template<RoundingMode mode>
        KSIMD_API(batch_t) round(batch_t v) noexcept
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
                __m256d sign_bit = _mm256_set1_pd(SignBitMask<float32>);
                // __m256d sign_mask = _mm256_and_pd(v.v[I], sign_bit);

                // 构造一个具有相同符号的0.5
                // __m256d half = _mm256_or_pd(_mm256_set1_pd(0x1.0p-1f), _mm256_and_pd(v.v[I], sign_bit));

                return { _mm256_round_pd(_mm256_add_pd(v.v[I],
                    _mm256_or_pd(_mm256_set1_pd(0x1.0p-1f), _mm256_and_pd(v.v[I], sign_bit))),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return { _mm256_round_pd(v.v[I], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
        }

        KSIMD_API(batch_t) abs(batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(SignBitClearMask<float64>);
            return { _mm256_and_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) neg(batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(SignBitMask<float64>);
            return { _mm256_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_min_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_max_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_EQ_OQ)... };
        }

        KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NEQ_UQ)... };
        }

        KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_GT_OQ)... };
        }

        KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NGT_UQ)... };
        }

        KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_GE_OQ)... };
        }

        KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NGE_UQ)... };
        }

        KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_LT_OQ)... };
        }

        KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NLT_UQ)... };
        }

        KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_LE_OQ)... };
        }

        KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_NLE_UQ)... };
        }

        KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_UNORD_Q)... };
        }

        KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            // __m256d l_nan = _mm256_cmp_pd(lhs.v[I], lhs.v[I], _CMP_UNORD_Q);
            // __m256d r_nan = _mm256_cmp_pd(rhs.v[I], rhs.v[I], _CMP_UNORD_Q);
            return { _mm256_and_pd(
                _mm256_cmp_pd(lhs.v[I], lhs.v[I], _CMP_UNORD_Q),
                _mm256_cmp_pd(rhs.v[I], rhs.v[I], _CMP_UNORD_Q))... };
        }

        KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_pd(lhs.v[I], rhs.v[I], _CMP_ORD_Q)... };
        }

        KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
            __m256d inf = _mm256_set1_pd(Inf<float64>);

            // __m256d combined = _mm256_and_pd(lhs.v[I], rhs.v[I]);

            return { _mm256_cmp_pd(_mm256_and_pd(_mm256_and_pd(lhs.v[I], rhs.v[I]), abs_mask), inf, _CMP_LT_OQ)... };
        }

        KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<float64>);
            __m256d inf = _mm256_set1_pd(Inf<float64>);

            // __m256d l_finite = _mm256_cmp_pd(_mm256_and_pd(lhs.v[I], abs_mask), inf, _CMP_LT_OQ);
            // __m256d r_finite = _mm256_cmp_pd(_mm256_and_pd(rhs.v[I], abs_mask), inf, _CMP_LT_OQ);

            return { _mm256_and_pd(
                _mm256_cmp_pd(_mm256_and_pd(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                _mm256_cmp_pd(_mm256_and_pd(rhs.v[I], abs_mask), inf, _CMP_LT_OQ))... };
        }

        KSIMD_API(batch_t) bit_not(batch_t v) noexcept
        {
            __m256d mask = _mm256_set1_pd(OneBlock<float64>);
            return { _mm256_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_and_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_andnot_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_or_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_xor_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm256_or_pd(_mm256_and_pd(mask.v[I], a.v[I]), _mm256_andnot_pd(mask.v[I], b.v[I]))... };
        }

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm256_blendv_pd(b.v[I], a.v[I], mask.m[I])... };
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_AVX_float64 = Executor_AVX_Impl_float64<std::make_index_sequence<reg_count>>;

    // AVX2
    template<typename = void>
    struct Executor_AVX2_Impl_float64;

    template<size_t... I>
    struct Executor_AVX2_Impl_float64<std::index_sequence<I...>>
        : Executor_AVX_Impl_float64<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_AVX2_float64 = Executor_AVX2_Impl_float64<std::make_index_sequence<reg_count>>;

    // AVX2_FMA3
    template<typename = void>
    struct Executor_AVX2_FMA3_Impl_float64;

    #define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_AVX2_FMA3_Impl_float64<std::index_sequence<I...>>
        : Executor_AVX2_Impl_float64<std::index_sequence<I...>>
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_AVX_Family<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float64, sizeof...(I)>)

    private:
        using base = Executor_AVX2_Impl_float64<std::index_sequence<I...>>;

    public:
        using base::sequence;

        KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
        {
            __m256d stride_v = _mm256_set1_pd(stride);
            __m256d base_v = _mm256_set1_pd(base);
            __m256d iota = _mm256_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm256_fmadd_pd(stride_v, iota, base_v))... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm256_fmadd_pd(a.v[I], b.v[I], c.v[I])... };
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_AVX2_FMA3_float64 = Executor_AVX2_FMA3_Impl_float64<std::make_index_sequence<reg_count>>;
}

// -------------------------------- operators --------------------------------
#define KSIMD_API(...) KSIMD_OP_AVX_API __VA_ARGS__ KSIMD_CALL_CONV
namespace x86_vector256
{
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator+(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::add(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::sub(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator*(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::mul(lhs, rhs);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator/(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::div(lhs, rhs);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> v) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::neg(v);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator&(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::bit_and(lhs, rhs);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator|(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::bit_or(lhs, rhs);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator^(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::bit_xor(lhs, rhs);
    }
    
    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator~(Batch<float64, reg_count> v) noexcept
    {
        return detail::Executor_AVX_float64<reg_count>::bit_not(v);
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
#undef KSIMD_API

// base op mixin
#define KSIMD_BATCH_T x86_vector256::Batch<float64, 1>
namespace detail
{
    #define KSIMD_API(...) KSIMD_OP_AVX_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_AVX_float64
    {
        KSIMD_API(float64) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            __m128d low = _mm256_castpd256_pd128(v.v[0]);
            __m128d high = _mm256_extractf128_pd(v.v[0], 0b1);
            __m128d sum128 = _mm_add_pd(low, high);

            // [2+4, 1+3] + [?, 2+4] = [1+2+3+4]
            __m128d sum64 = _mm_add_pd(sum128, _mm_unpackhi_pd(sum128, sum128));

            return _mm_cvtsd_f64(sum64);
        }
    };
    #undef KSIMD_API
}
#undef KSIMD_BATCH_T

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float64>
    : detail::Executor_AVX_float64<1>
    , detail::Base_Mixin_AVX_float64
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float64>
    : detail::Executor_AVX2_float64<1>
    , detail::Base_Mixin_AVX_float64
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float64>
    : detail::Executor_AVX2_FMA3_float64<1>
    , detail::Base_Mixin_AVX_float64
{};

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

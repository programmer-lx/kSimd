#pragma once

#include <utility> // std::index_sequence

#include "traits.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 1.0, 0.0

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // SSE2
    template<typename = void>
    struct Executor_SSE2_Impl_float64;

    #define KSIMD_API(ret) KSIMD_OP_SSE2_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_SSE2_Impl_float64<std::index_sequence<I...>>
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE2_Plus<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float64, sizeof...(I)>)
        
        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float64* mem, mask_t mask) noexcept
        {
            (_mm_store_pd(&mem[I * RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(mask_t) test_load_mask(const float64* mem) noexcept
        {
            return { _mm_load_pd(&mem[I * RegLanes])... };
        }
        #endif

        KSIMD_API(batch_t) load(const float64* mem) noexcept
        {
            return { _mm_load_pd(&mem[I * RegLanes])... };
        }

        KSIMD_API(batch_t) loadu(const float64* mem) noexcept
        {
            return { _mm_loadu_pd(&mem[I * RegLanes])... };
        }

        KSIMD_API(batch_t) load_partial(const float64* mem, size_t count) noexcept
        {
            count = count > TotalLanes ? TotalLanes : count;

            if (count == 0)
                return zero();

            batch_t res = zero();
            std::memcpy(res.v, mem, sizeof(float64) * count);
            return res;
        }

        KSIMD_API(void) store(float64* mem, batch_t v) noexcept
        {
            (_mm_store_pd(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float64* mem, batch_t v) noexcept
        {
            (_mm_storeu_pd(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) store_partial(float64* mem, batch_t v, size_t count) noexcept
        {
            count = count > TotalLanes ? TotalLanes : count;
            if (count == 0)
                return;

            std::memcpy(mem, v.v, sizeof(float64) * count);
        }

        KSIMD_API(batch_t) undefined() noexcept
        {
            return { ((void)I, _mm_undefined_pd())... };
        }

        KSIMD_API(batch_t) zero() noexcept
        {
            return { ((void)I, _mm_setzero_pd())... };
        }

        KSIMD_API(batch_t) set(float64 x) noexcept
        {
            return { ((void)I, _mm_set1_pd(x))... };
        }
        
        KSIMD_API(batch_t) sequence() noexcept
        {
            __m128d iota = _mm_set_pd(KSIMD_IOTA);
            return { ((void)I, iota)... };
        }

        KSIMD_API(batch_t) sequence(float64 base) noexcept
        {
            __m128d base_v = _mm_set1_pd(base);
            __m128d iota = _mm_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm_add_pd(iota, base_v))... };
        }

        KSIMD_API(batch_t) sequence(float64 base, float64 stride) noexcept
        {
            __m128d stride_v = _mm_set1_pd(stride);
            __m128d base_v = _mm_set1_pd(base);
            __m128d iota = _mm_set_pd(KSIMD_IOTA);
            return { ((void)I, _mm_add_pd(_mm_mul_pd(stride_v, iota), base_v))... };
        }

        KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_add_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_sub_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_mul_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_div_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) one_div(batch_t v) noexcept
        {
            return { _mm_div_pd(_mm_set1_pd(1.0), v.v[I])... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm_add_pd(_mm_mul_pd(a.v[I], b.v[I]), c.v[I])... };
        }

        KSIMD_API(batch_t) sqrt(batch_t v) noexcept
        {
            return { _mm_sqrt_pd(v.v[I])... };
        }

        KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
        {
            return { _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(v.v[I]))... };
        }

        KSIMD_API(batch_t) abs(batch_t v) noexcept
        {
            __m128d mask = _mm_set1_pd(SignBitClearMask<float64>);
            return { _mm_and_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) neg(batch_t v) noexcept
        {
            __m128d mask = _mm_set1_pd(SignBitMask<float64>);
            return { _mm_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_min_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_max_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpeq_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpneq_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpgt_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpngt_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpge_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnge_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmplt_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnlt_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmple_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnle_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpunord_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            // __m128d l_nan = _mm_cmpunord_pd(lhs.v[I], lhs.v[I]);
            // __m128d r_nan = _mm_cmpunord_pd(rhs.v[I], rhs.v[I]);
            return { _mm_and_pd(_mm_cmpunord_pd(lhs.v[I], lhs.v[I]), _mm_cmpunord_pd(rhs.v[I], rhs.v[I]))... };
        }

        KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpord_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m128d abs_mask = _mm_set1_pd(SignBitClearMask<float64>);
            __m128d inf = _mm_set1_pd(Inf<float64>);
            return {
                _mm_or_pd(
                    _mm_cmplt_pd(_mm_and_pd(lhs.v[I], abs_mask), inf),
                    _mm_cmplt_pd(_mm_and_pd(rhs.v[I], abs_mask), inf)
                )...
            };
        }

        KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m128d abs_mask = _mm_set1_pd(SignBitClearMask<float64>);
            __m128d inf = _mm_set1_pd(Inf<float64>);

            // __m128d l_finite = _mm_cmplt_pd(_mm_and_pd(lhs.v[I], abs_mask), inf);
            // __m128d r_finite = _mm_cmplt_pd(_mm_and_pd(rhs.v[I], abs_mask), inf);

            return {
                _mm_and_pd(
                    _mm_cmplt_pd(_mm_and_pd(lhs.v[I], abs_mask), inf),
                    _mm_cmplt_pd(_mm_and_pd(rhs.v[I], abs_mask), inf)
                )...
            };
        }

        KSIMD_API(batch_t) bit_not(batch_t v) noexcept
        {
            __m128d mask = _mm_set1_pd(OneBlock<float64>);
            return { _mm_xor_pd(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_and_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_andnot_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_or_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_xor_pd(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_or_pd(_mm_and_pd(mask.v[I], a.v[I]), _mm_andnot_pd(mask.v[I], b.v[I]))... };
        }

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_or_pd(_mm_and_pd(mask.m[I], a.v[I]), _mm_andnot_pd(mask.m[I], b.v[I]))... };
        }
    };
    #undef KSIMD_API
    
    template<size_t reg_count>
    using Executor_SSE2_float64 = Executor_SSE2_Impl_float64<std::make_index_sequence<reg_count>>;

    // SSE3
    template<typename = void>
    struct Executor_SSE3_Impl_float64;

    template<size_t... I>
    struct Executor_SSE3_Impl_float64<std::index_sequence<I...>>
        : Executor_SSE2_Impl_float64<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_SSE3_float64 = Executor_SSE3_Impl_float64<std::make_index_sequence<reg_count>>;

    // SSSE3
    template<typename = void>
    struct Executor_SSSE3_Impl_float64;

    template<size_t... I>
    struct Executor_SSSE3_Impl_float64<std::index_sequence<I...>>
        : Executor_SSE3_Impl_float64<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_SSSE3_float64 = Executor_SSSE3_Impl_float64<std::make_index_sequence<reg_count>>;

    // SSE4.1
    template<typename = void>
    struct Executor_SSE4_1_Impl_float64;

    #define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_SSE4_1_Impl_float64<std::index_sequence<I...>>
        : Executor_SSE3_Impl_float64<std::index_sequence<I...>>
        , BaseOpHelper
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE2_Plus<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float64, sizeof...(I)>)

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_blendv_pd(b.v[I], a.v[I], mask.m[I])... };
        }

        template<RoundingMode mode>
        KSIMD_API(batch_t) round(batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return { _mm_round_pd(v.v[I], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return { _mm_round_pd(v.v[I], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return { _mm_round_pd(v.v[I], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
                __m128d sign_bit = _mm_set1_pd(SignBitMask<float32>);
                // __m128d sign_mask = _mm_and_pd(v.v[I], sign_bit);

                // 构造一个具有相同符号的0.5 (0x1.0p-1f == 0.5f 16进制精确表示)
                // __m128d half = _mm_or_pd(_mm_set1_pd(0x1.0p-1f), _mm_and_pd(v.v[I], sign_bit));

                return {
                    _mm_round_pd(_mm_add_pd(v.v[I], _mm_or_pd(_mm_set1_pd(0x1.0p-1f), _mm_and_pd(v.v[I], sign_bit))),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)...
                };
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return { _mm_round_pd(v.v[I], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_SSE4_1_float64 = Executor_SSE4_1_Impl_float64<std::make_index_sequence<reg_count>>;
}

// -------------------------------- operators --------------------------------
#define KSIMD_API(...) KSIMD_OP_SSE2_API static __VA_ARGS__ KSIMD_CALL_CONV
namespace x86_vector128
{
#define KSIMD_BATCH_T Batch<float64, reg_count>

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator+(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::add(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::sub(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator*(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::mul(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator/(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::div(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator-(Batch<float64, reg_count> v) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::neg(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator&(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::bit_and(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator|(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::bit_or(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator^(Batch<float64, reg_count> lhs, Batch<float64, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::bit_xor(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float64, reg_count>) operator~(Batch<float64, reg_count> v) noexcept
    {
        return detail::Executor_SSE2_float64<reg_count>::bit_not(v);
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

// base op mixin
#define KSIMD_BATCH_T x86_vector128::Batch<float64, 1>
namespace detail
{
    #define KSIMD_API(...) KSIMD_OP_SSE2_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_SSE2_float64
    {
        KSIMD_API(float64) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            __m128d sum64 = _mm_add_pd(v.v[0], _mm_shuffle_pd(v.v[0], v.v[0], _MM_SHUFFLE2(0, 1)));
            return _mm_cvtsd_f64(sum64);
        }
    };
    #undef KSIMD_API

    #define KSIMD_API(...) KSIMD_OP_SSE3_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_SSE3_float64
    {
        KSIMD_API(float64) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            __m128d sum64 = _mm_add_pd(v.v[0], _mm_shuffle_pd(v.v[0], v.v[0], _MM_SHUFFLE2(0, 1)));
            return _mm_cvtsd_f64(sum64);
        }
    };
    #undef KSIMD_API
}
#undef KSIMD_BATCH_T

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

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float64>
    : detail::Executor_SSE2_float64<1>
    , detail::Base_Mixin_SSE2_float64
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float64>
    : detail::Executor_SSE3_float64<1>
    , detail::Base_Mixin_SSE3_float64
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float64>
    : detail::Executor_SSSE3_float64<1>
    , detail::Base_Mixin_SSE3_float64
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float64>
    : detail::Executor_SSE4_1_float64<1>
    , detail::Base_Mixin_SSE3_float64
{};

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

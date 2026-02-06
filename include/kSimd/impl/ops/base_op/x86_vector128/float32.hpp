#pragma once

#include <utility> // std::index_sequence

#include "traits.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector128.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 3.f, 2.f, 1.f, 0.f

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // SSE
    template<typename = void>
    struct Executor_SSE_Impl_float32;

    #define KSIMD_API(ret) KSIMD_OP_SSE_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_SSE_Impl_float32<std::index_sequence<I...>>
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float32, sizeof...(I)>)

        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float32* mem, mask_t mask) noexcept
        {
            (_mm_store_ps(&mem[I * RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(mask_t) test_load_mask(const float32* mem) noexcept
        {
            return { (_mm_load_ps(&mem[I * RegLanes]))... };
        }
        #endif

        KSIMD_API(batch_t) load(const float32* mem) noexcept
        {
            return { (_mm_load_ps(&mem[I * RegLanes]))... };
        }

        KSIMD_API(batch_t) loadu(const float32* mem) noexcept
        {
            return { (_mm_loadu_ps(&mem[I * RegLanes]))... };
        }

        KSIMD_API(void) store(float32* mem, batch_t v) noexcept
        {
            (_mm_store_ps(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float32* mem, batch_t v) noexcept
        {
            (_mm_storeu_ps(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(batch_t) undefined() noexcept
        {
            return { ((void)I, _mm_undefined_ps())... };
        }

        KSIMD_API(batch_t) zero() noexcept
        {
            return { ((void)I, _mm_setzero_ps())... };
        }

        KSIMD_API(batch_t) set(float32 x) noexcept
        {
            return { ((void)I, _mm_set1_ps(x))... };
        }

        KSIMD_API(batch_t) sequence() noexcept
        {
            __m128 iota = _mm_set_ps(KSIMD_IOTA);
            return { ((void)I, iota)... };
        }

        KSIMD_API(batch_t) sequence(float32 base) noexcept
        {
            __m128 iota = _mm_set_ps(KSIMD_IOTA);
            __m128 base_v = _mm_set1_ps(base);
            return { ((void)I, _mm_add_ps(iota, base_v))... };
        }

        KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
        {
            __m128 iota = _mm_set_ps(KSIMD_IOTA);
            __m128 stride_v = _mm_set1_ps(stride);
            __m128 base_v = _mm_set1_ps(base);
            return { ((void)I, _mm_add_ps(_mm_mul_ps(stride_v, iota), base_v))... };
        }

        KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_add_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_sub_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_mul_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_div_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) one_div(batch_t v) noexcept
        {
            return { _mm_rcp_ps(v.v[I])... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm_add_ps(_mm_mul_ps(a.v[I], b.v[I]), c.v[I])... };
        }

        KSIMD_API(batch_t) sqrt(batch_t v) noexcept
        {
            return { _mm_sqrt_ps(v.v[I])... };
        }

        KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
        {
            return { _mm_rsqrt_ps(v.v[I])... };
        }

        KSIMD_API(batch_t) abs(batch_t v) noexcept
        {
            return { _mm_and_ps(v.v[I], _mm_set1_ps(SignBitClearMask<float32>))... };
        }

        KSIMD_API(batch_t) neg(batch_t v) noexcept
        {
            return { _mm_xor_ps(v.v[I], _mm_set1_ps(SignBitMask<float32>))... };
        }

        KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_min_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_max_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpeq_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpneq_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpgt_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpngt_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpge_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnge_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmplt_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnlt_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmple_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpnle_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpunord_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_and_ps(_mm_cmpunord_ps(lhs.v[I], lhs.v[I]), _mm_cmpunord_ps(rhs.v[I], rhs.v[I]))... };
        }

        KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_cmpord_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float32>);
            __m128 inf = _mm_set1_ps(Inf<float32>);
            return {
                _mm_or_ps(
                    _mm_cmplt_ps(_mm_and_ps(lhs.v[I], abs_mask), inf),
                    _mm_cmplt_ps(_mm_and_ps(rhs.v[I], abs_mask), inf)
                )...
            };
        }

        KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float32>);
            __m128 inf = _mm_set1_ps(Inf<float32>);

            // __m128 l_finite = _mm_cmplt_ps(_mm_and_ps(lhs.v[I], abs_mask), inf);
            // __m128 r_finite = _mm_cmplt_ps(_mm_and_ps(rhs.v[I], abs_mask), inf);

            return {
                _mm_and_ps(
                    _mm_cmplt_ps(_mm_and_ps(lhs.v[I], abs_mask), inf),
                    _mm_cmplt_ps(_mm_and_ps(rhs.v[I], abs_mask), inf)
                )...
            };
        }

        KSIMD_API(batch_t) bit_not(batch_t v) noexcept
        {
            __m128 mask = _mm_set1_ps(OneBlock<float32>);
            return { _mm_xor_ps(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_and_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_andnot_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_or_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm_xor_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_or_ps(_mm_and_ps(mask.v[I], a.v[I]), _mm_andnot_ps(mask.v[I], b.v[I]))... };
        }

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_or_ps(_mm_and_ps(mask.m[I], a.v[I]), _mm_andnot_ps(mask.m[I], b.v[I]))... };
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_SSE_float32 = Executor_SSE_Impl_float32<std::make_index_sequence<reg_count>>;

    // SSE2
    template<typename = void>
    struct Executor_SSE2_Impl_float32;

    template<size_t... I>
    struct Executor_SSE2_Impl_float32<std::index_sequence<I...>>
        : Executor_SSE_Impl_float32<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_SSE2_float32 = Executor_SSE2_Impl_float32<std::make_index_sequence<reg_count>>;

    // SSE3
    template<typename = void>
    struct Executor_SSE3_Impl_float32;

    template<size_t... I>
    struct Executor_SSE3_Impl_float32<std::index_sequence<I...>>
        : Executor_SSE2_Impl_float32<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_SSE3_float32 = Executor_SSE3_Impl_float32<std::make_index_sequence<reg_count>>;

    // SSSE3
    template<typename = void>
    struct Executor_SSSE3_Impl_float32;

    template<size_t... I>
    struct Executor_SSSE3_Impl_float32<std::index_sequence<I...>>
        : Executor_SSE3_Impl_float32<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_SSSE3_float32 = Executor_SSSE3_Impl_float32<std::make_index_sequence<reg_count>>;

    // SSE4.1
    template<typename = void>
    struct Executor_SSE4_1_Impl_float32;

    #define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_SSE4_1_Impl_float32<std::index_sequence<I...>>
        : Executor_SSSE3_Impl_float32<std::index_sequence<I...>>
        , BaseOpHelper
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_SSE2_Plus<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, sizeof...(I)>)

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm_blendv_ps(b.v[I], a.v[I], mask.m[I])... };
        }

        template<RoundingMode mode>
        KSIMD_API(batch_t) round(batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return { _mm_round_ps(v.v[I], _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return { _mm_round_ps(v.v[I], _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return { _mm_round_ps(v.v[I], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)... };
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                // 提取符号位，如果v是负数，则sign_mask为0b1000...，如果v是正数，则sign_mask为0b0000...
                __m128 sign_bit = _mm_set1_ps(SignBitMask<float32>);
                // __m128 sign_mask = _mm_and_ps(v.v[I], sign_bit);

                // 构造一个具有相同符号的0.5 (0x1.0p-1f == 0.5f 16进制精确表示)
                // __m128 half = _mm_or_ps(_mm_set1_ps(0x1.0p-1f), _mm_and_ps(v.v[I], sign_bit));

                return {
                    _mm_round_ps(_mm_add_ps(v.v[I], _mm_or_ps(_mm_set1_ps(0x1.0p-1f), _mm_and_ps(v.v[I], sign_bit))),
                        _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)...
                };
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return { _mm_round_ps(v.v[I], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)... };
            }
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_SSE4_1_float32 = Executor_SSE4_1_Impl_float32<std::make_index_sequence<reg_count>>;
} // namespace detail

// -------------------------------- operators --------------------------------
#define KSIMD_API(...) KSIMD_OP_SSE_API static __VA_ARGS__ KSIMD_CALL_CONV
namespace x86_vector128
{
    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator+(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::add(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::sub(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator*(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::mul(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator/(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::div(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> v) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::neg(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator&(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::bit_and(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator|(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::bit_or(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator^(Batch<float32, reg_count> lhs,
                                                   Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::bit_xor(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator~(Batch<float32, reg_count> v) noexcept
    {
        return detail::Executor_SSE_float32<reg_count>::bit_not(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator+=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator-=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator*=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator/=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator&=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator|=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>&) operator^=(Batch<float32, reg_count>& lhs,
                                                     Batch<float32, reg_count> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace x86_vector128
#undef KSIMD_API


// base op mixin
#define KSIMD_BATCH_T x86_vector128::Batch<float32, 1>
namespace detail
{
    #define KSIMD_API(...) KSIMD_OP_SSE_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_SSE_float32
    {
        KSIMD_API(float32) reduce_add(KSIMD_BATCH_T v) noexcept
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
            __m128 t1 = _mm_shuffle_ps(v.v[0], v.v[0], _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sum64 = _mm_add_ps(v.v[0], t1);
            __m128 t2 = _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 0, 3, 2));
            __m128 sum32 = _mm_add_ps(sum64, t2);
            return _mm_cvtss_f32(sum32);
        }
    };
    #undef KSIMD_API

    #define KSIMD_API(...) KSIMD_OP_SSE3_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_SSE3_float32
    {
        KSIMD_API(float32) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            __m128 result = _mm_hadd_ps(v.v[0], v.v[0]);
            result = _mm_hadd_ps(result, result);
            return _mm_cvtss_f32(result);
        }
    };
    #undef KSIMD_API
}
#undef KSIMD_BATCH_T

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float32>
    : detail::Executor_SSE_float32<1>
    , detail::Base_Mixin_SSE_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float32>
    : detail::Executor_SSE2_float32<1>
    , detail::Base_Mixin_SSE_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float32>
    : detail::Executor_SSE3_float32<1>
    , detail::Base_Mixin_SSE3_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float32>
    : detail::Executor_SSSE3_float32<1>
    , detail::Base_Mixin_SSE3_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32>
    : detail::Executor_SSE4_1_float32<1>
    , detail::Base_Mixin_SSE3_float32
{};

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

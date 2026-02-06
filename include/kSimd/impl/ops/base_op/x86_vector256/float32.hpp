#pragma once

#include "traits.hpp"
#include "kSimd/impl/ops/vector_types/x86_vector256.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/number.hpp"

#define KSIMD_IOTA 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    // AVX
    template<typename = void>
    struct Executor_AVX_Impl_float32;

    #define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_AVX_Impl_float32<std::index_sequence<I...>>
        : BaseOpHelper
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_AVX_Family<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float32, sizeof...(I)>)

        #if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(float32* mem, mask_t mask) noexcept
        {
            (_mm256_store_ps(&mem[I * RegLanes], mask.m[I]), ...);
        }
        KSIMD_API(mask_t) test_load_mask(const float32* mem) noexcept
        {
            return { _mm256_load_ps(&mem[I * RegLanes])... };
        }
        #endif

        KSIMD_API(mask_t) mask_from_lanes(size_t count) noexcept
        {
            __m256 cnt = _mm256_set1_ps(static_cast<float32>(count));
            __m256 idx = _mm256_set_ps(KSIMD_IOTA);
            return { ((void)I, _mm256_cmp_ps(idx, cnt, _CMP_LT_OQ))... };
        }

        KSIMD_API(batch_t) load(const float32* mem) noexcept
        {
            return { _mm256_load_ps(&mem[I * RegLanes])... };
        }

        KSIMD_API(batch_t) loadu(const float32* mem) noexcept
        {
            return { _mm256_loadu_ps(&mem[I * RegLanes])... };
        }

        KSIMD_API(void) store(float32* mem, batch_t v) noexcept
        {
            (_mm256_store_ps(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(void) storeu(float32* mem, batch_t v) noexcept
        {
            (_mm256_storeu_ps(&mem[I * RegLanes], v.v[I]), ...);
        }

        KSIMD_API(batch_t) mask_load(const float32* mem, mask_t mask) noexcept
        {
            return { _mm256_maskload_ps(mem, _mm256_castps_si256(mask.m[I]))... };
        }

        KSIMD_API(batch_t) mask_load(const float32* mem, mask_t mask, batch_t default_value) noexcept
        {
            batch_t loaded = mask_load(mem, mask);
            return { _mm256_or_ps(loaded.v[I], _mm256_andnot_ps(mask.m[I], default_value.v[I]))... };
        }

        KSIMD_API(batch_t) mask_loadu(const float32* mem, mask_t mask) noexcept
        {
            return mask_load(mem, mask);
        }

        KSIMD_API(batch_t) mask_loadu(const float32* mem, mask_t mask, batch_t default_value) noexcept
        {
            return mask_load(mem, mask, default_value);
        }

        KSIMD_API(void) mask_store(float32* mem, batch_t v, mask_t mask) noexcept
        {
            (_mm256_maskstore_ps(mem, _mm256_castps_si256(mask.m[I]), v.v[I]), ...);
        }

        KSIMD_API(void) mask_storeu(float32* mem, batch_t v, mask_t mask) noexcept
        {
            (_mm256_maskstore_ps(mem, _mm256_castps_si256(mask.m[I]), v.v[I]), ...);
        }

        KSIMD_API(batch_t) undefined() noexcept
        {
            return { ((void)I, _mm256_undefined_ps())... };
        }

        KSIMD_API(batch_t) zero() noexcept
        {
            return { ((void)I, _mm256_setzero_ps())... };
        }

        KSIMD_API(batch_t) set(float32 x) noexcept
        {
            return { ((void)I, _mm256_set1_ps(x))... };
        }

        KSIMD_API(batch_t) sequence() noexcept
        {
            __m256 iota = _mm256_set_ps(KSIMD_IOTA);
            return { ((void)I, iota)... };
        }

        KSIMD_API(batch_t) sequence(float32 base) noexcept
        {
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(KSIMD_IOTA);
            return { ((void)I, _mm256_add_ps(iota, base_v))... };
        }

        KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
        {
            __m256 stride_v = _mm256_set1_ps(stride);
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(KSIMD_IOTA);
            return { ((void)I, _mm256_add_ps(_mm256_mul_ps(stride_v, iota), base_v))... };
        }

        KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_add_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_sub_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_mul_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_div_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) one_div(batch_t v) noexcept
        {
            return { _mm256_rcp_ps(v.v[I])... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm256_add_ps(_mm256_mul_ps(a.v[I], b.v[I]), c.v[I])... };
        }

        KSIMD_API(batch_t) sqrt(batch_t v) noexcept
        {
            return { _mm256_sqrt_ps(v.v[I])... };
        }

        KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
        {
            return { _mm256_rsqrt_ps(v.v[I])... };
        }

        template<RoundingMode mode>
        KSIMD_API(batch_t) round(batch_t v) noexcept
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

        KSIMD_API(batch_t) abs(batch_t v) noexcept
        {
            // 将 sign bit 反转即可
            return { _mm256_and_ps(v.v[I], _mm256_set1_ps(SignBitClearMask<float32>))... };
        }

        KSIMD_API(batch_t) neg(batch_t v) noexcept
        {
            __m256 mask = _mm256_set1_ps(SignBitMask<float32>);
            return { _mm256_xor_ps(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_min_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_max_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_EQ_OQ)... };
        }

        KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NEQ_UQ)... };
        }

        KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_GT_OQ)... };
        }

        KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NGT_UQ)... };
        }

        KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_GE_OQ)... };
        }

        KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NGE_UQ)... };
        }

        KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_LT_OQ)... };
        }

        KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NLT_UQ)... };
        }

        KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_LE_OQ)... };
        }

        KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_NLE_UQ)... };
        }

        KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_UNORD_Q)... };
        }

        KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            // __m256 l_nan = _mm256_cmp_ps(lhs.v[I], lhs.v[I], _CMP_UNORD_Q);
            // __m256 r_nan = _mm256_cmp_ps(rhs.v[I], rhs.v[I], _CMP_UNORD_Q);
            return { _mm256_and_ps(
                _mm256_cmp_ps(lhs.v[I], lhs.v[I], _CMP_UNORD_Q),
                _mm256_cmp_ps(rhs.v[I], rhs.v[I], _CMP_UNORD_Q))... };
        }

        KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_cmp_ps(lhs.v[I], rhs.v[I], _CMP_ORD_Q)... };
        }

        KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
            __m256 inf = _mm256_set1_ps(Inf<float32>);

            // __m256 combined = _mm256_and_ps(lhs.v[I], rhs.v[I]);

            return { _mm256_cmp_ps(_mm256_and_ps(_mm256_and_ps(lhs.v[I], rhs.v[I]), abs_mask), inf, _CMP_LT_OQ)... };
        }

        KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
        {
            __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
            __m256 inf = _mm256_set1_ps(Inf<float32>);

            // __m256 l_finite = _mm256_cmp_ps(_mm256_and_ps(lhs.v[I], abs_mask), inf, _CMP_LT_OQ);
            // __m256 r_finite = _mm256_cmp_ps(_mm256_and_ps(rhs.v[I], abs_mask), inf, _CMP_LT_OQ);

            return { _mm256_and_ps(
                _mm256_cmp_ps(_mm256_and_ps(lhs.v[I], abs_mask), inf, _CMP_LT_OQ),
                _mm256_cmp_ps(_mm256_and_ps(rhs.v[I], abs_mask), inf, _CMP_LT_OQ))... };
        }

        KSIMD_API(batch_t) bit_not(batch_t v) noexcept
        {
            __m256 mask = _mm256_set1_ps(OneBlock<float32>);
            return { _mm256_xor_ps(v.v[I], mask)... };
        }

        KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_and_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_andnot_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_or_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
        {
            return { _mm256_xor_ps(lhs.v[I], rhs.v[I])... };
        }

        KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm256_or_ps(_mm256_and_ps(mask.v[I], a.v[I]), _mm256_andnot_ps(mask.v[I], b.v[I]))... };
        }

        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            return { _mm256_blendv_ps(b.v[I], a.v[I], mask.m[I])... };
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_AVX_float32 = Executor_AVX_Impl_float32<std::make_index_sequence<reg_count>>;

    // AVX2
    template<typename = void>
    struct Executor_AVX2_Impl_float32;

    template<size_t... I>
    struct Executor_AVX2_Impl_float32<std::index_sequence<I...>>
        : Executor_AVX_Impl_float32<std::index_sequence<I...>>
    {};

    template<size_t reg_count>
    using Executor_AVX2_float32 = Executor_AVX2_Impl_float32<std::make_index_sequence<reg_count>>;

    // AVX2_FMA3
    template<typename = void>
    struct Executor_AVX2_FMA3_Impl_float32;

    #define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_API static ret KSIMD_CALL_CONV
    template<size_t... I>
    struct Executor_AVX2_FMA3_Impl_float32<std::index_sequence<I...>>
        : Executor_AVX2_Impl_float32<std::index_sequence<I...>>
    {
        KSIMD_DETAIL_TRAITS(BaseOpTraits_AVX_Family<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, sizeof...(I)>)

    private:
        using base = Executor_AVX2_Impl_float32<std::index_sequence<I...>>;

    public:
        using base::sequence;

        KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
        {
            __m256 stride_v = _mm256_set1_ps(stride);
            __m256 base_v = _mm256_set1_ps(base);
            __m256 iota = _mm256_set_ps(KSIMD_IOTA);
            return { ((void)I, _mm256_fmadd_ps(stride_v, iota, base_v))... };
        }

        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            return { _mm256_fmadd_ps(a.v[I], b.v[I], c.v[I])... };
        }
    };
    #undef KSIMD_API

    template<size_t reg_count>
    using Executor_AVX2_FMA3_float32 = Executor_AVX2_FMA3_Impl_float32<std::make_index_sequence<reg_count>>;
}

// -------------------------------- operators --------------------------------
#define KSIMD_API(...) KSIMD_OP_AVX_API __VA_ARGS__ KSIMD_CALL_CONV
namespace x86_vector256
{
    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator+(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::add(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::sub(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator*(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::mul(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator/(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::div(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator-(Batch<float32, reg_count> v) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::neg(v);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator&(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::bit_and(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator|(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::bit_or(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator^(Batch<float32, reg_count> lhs, Batch<float32, reg_count> rhs) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::bit_xor(lhs, rhs);
    }

    template<size_t reg_count>
    KSIMD_API(Batch<float32, reg_count>) operator~(Batch<float32, reg_count> v) noexcept
    {
        return detail::Executor_AVX_float32<reg_count>::bit_not(v);
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
#undef KSIMD_API

// base op mixin
#define KSIMD_BATCH_T x86_vector256::Batch<float32, 1>
namespace detail
{
    #define KSIMD_API(...) KSIMD_OP_AVX_API static __VA_ARGS__ KSIMD_CALL_CONV
    struct Base_Mixin_AVX_float32
    {
        KSIMD_API(float32) reduce_add(KSIMD_BATCH_T v) noexcept
        {
            // [8,7,6,5] + [4,3,2,1] = [8+4, 7+3, 6+2, 5+1]
            __m128 low = _mm256_castps256_ps128(v.v[0]);
            __m128 high = _mm256_extractf128_ps(v.v[0], 1);
            __m128 sum128 = _mm_add_ps(low, high);

            // [8+4, 7+3, 8+4, 7+3] + [8+4, 7+3, 6+2, 5+1] = [?, ?, 2468, 1357]
            __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));

            // [?, ?, 2468, 1357] + [?, ?, 2468, 2468] = [?, ?, ?, 12345678]
            __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(1, 1, 1, 1)));

            return _mm_cvtss_f32(sum32);
        }
    };
    #undef KSIMD_API
}
#undef KSIMD_BATCH_T

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float32>
    : detail::Executor_AVX_float32<1>
    , detail::Base_Mixin_AVX_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float32>
    : detail::Executor_AVX2_float32<1>
    , detail::Base_Mixin_AVX_float32
{};

template<>
struct BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32>
    : detail::Executor_AVX2_FMA3_float32<1>
    , detail::Base_Mixin_AVX_float32
{};

KSIMD_NAMESPACE_END

#undef KSIMD_IOTA

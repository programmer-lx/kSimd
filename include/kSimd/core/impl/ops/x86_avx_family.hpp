// do not use include guard

#include "kSimd/IDE/IDE_hint.hpp"

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> // AVX+

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/traits.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        // --- types ---
        template<is_scalar_type>
        struct Batch
        {
            __m256i v;
        };

        template<>
        struct Batch<float32>
        {
            __m256 v;
        };

        template<>
        struct Batch<float64>
        {
            __m256d v;
        };

        template<is_scalar_type>
        struct Mask
        {
            __m256i m;
        };

        template<>
        struct Mask<float32>
        {
            __m256 m;
        };

        template<>
        struct Mask<float64>
        {
            __m256d m;
        };

        namespace detail
        {
            template<is_scalar_type S>
            struct op_impl;

            template<>
            struct op_impl<float32>
                : OpHelper
                , OpInfo<float32, Batch<float32>, Mask<float32>, 32, 32>
            {
                KSIMD_API(batch_t) load(const scalar_t* mem) noexcept
                {
                    return { _mm256_load_ps(mem) };
                }

                KSIMD_API(batch_t) loadu(const scalar_t* mem) noexcept
                {
                    return { _mm256_loadu_ps(mem) };
                }

                KSIMD_API(batch_t) load_partial(const scalar_t* mem, size_t count) noexcept
                {
                    count = count > Lanes ? Lanes : count;

                    if (count == 0) [[unlikely]]
                        return zero();

                    batch_t res = zero();
                    std::memcpy(&res.v, mem, sizeof(scalar_t) * count);
                    return res;
                }

                KSIMD_API(void) store(scalar_t* mem, batch_t v) noexcept
                {
                    _mm256_store_ps(mem, v.v);
                }

                KSIMD_API(void) storeu(scalar_t* mem, batch_t v) noexcept
                {
                    _mm256_storeu_ps(mem, v.v);
                }

                KSIMD_API(void) store_partial(scalar_t* mem, batch_t v, size_t count) noexcept
                {
                    count = count > Lanes ? Lanes : count;
                    if (count == 0) [[unlikely]]
                        return;

                    std::memcpy(mem, &v.v, sizeof(scalar_t) * count);
                }

                KSIMD_API(batch_t) undefined() noexcept
                {
                    return { _mm256_undefined_ps() };
                }

                KSIMD_API(batch_t) zero() noexcept
                {
                    return { _mm256_setzero_ps() };
                }

                KSIMD_API(batch_t) set(scalar_t x) noexcept
                {
                    return { _mm256_set1_ps(x) };
                }

                KSIMD_API(batch_t) sequence() noexcept
                {
                    return { _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f) };
                }

                KSIMD_API(batch_t) sequence(scalar_t base) noexcept
                {
                    __m256 base_v = _mm256_set1_ps(base);
                    __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
                    return { _mm256_add_ps(iota, base_v) };
                }

                KSIMD_API(batch_t) sequence(scalar_t base, scalar_t stride) noexcept
                {
                    __m256 stride_v = _mm256_set1_ps(stride);
                    __m256 base_v = _mm256_set1_ps(base);
                    __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
                    return { _mm256_fmadd_ps(stride_v, iota, base_v) };
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

                KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
                {
                    return { _mm256_fmadd_ps(a.v, b.v, c.v) };
                }

                KSIMD_API(batch_t) sqrt(batch_t v) noexcept
                {
                    return { _mm256_sqrt_ps(v.v) };
                }

                KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
                {
                    return { _mm256_rsqrt_ps(v.v) };
                }

                template<RoundingMode mode>
                KSIMD_API(batch_t) round(batch_t v) noexcept
                {
                    if constexpr (mode == RoundingMode::Up)
                    {
                        return { _mm256_round_ps(v.v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC) };
                    }
                    else if constexpr (mode == RoundingMode::Down)
                    {
                        return { _mm256_round_ps(v.v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC) };
                    }
                    else if constexpr (mode == RoundingMode::Nearest)
                    {
                        return { _mm256_round_ps(v.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC) };
                    }
                    else if constexpr (mode == RoundingMode::ToZero)
                    {
                        return { _mm256_round_ps(v.v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
                    }
                    else /* if constexpr (mode == RoundingMode::Round) */
                    {
                        __m256 sign_bit = _mm256_set1_ps(SignBitMask<scalar_t>);
                        __m256 half = _mm256_set1_ps(0x1.0p-1f);
                        return { _mm256_round_ps(_mm256_add_ps(v.v, _mm256_or_ps(half, _mm256_and_ps(v.v, sign_bit))),
                                                 _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
                    }
                }

                KSIMD_API(batch_t) abs(batch_t v) noexcept
                {
                    return { _mm256_and_ps(v.v, _mm256_set1_ps(SignBitClearMask<scalar_t>)) };
                }

                KSIMD_API(batch_t) neg(batch_t v) noexcept
                {
                    __m256 mask = _mm256_set1_ps(SignBitMask<scalar_t>);
                    return { _mm256_xor_ps(v.v, mask) };
                }

                KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
                {
                    return { _mm256_min_ps(lhs.v, rhs.v) };
                }

                KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
                {
                    return { _mm256_max_ps(lhs.v, rhs.v) };
                }

                KSIMD_API(batch_t) bit_not(batch_t v) noexcept
                {
                    __m256 mask = _mm256_set1_ps(OneBlock<scalar_t>);
                    return { _mm256_xor_ps(v.v, mask) };
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

#if defined(KSIMD_IS_TESTING)
                KSIMD_API(void) test_store_mask(scalar_t* mem, mask_t mask) noexcept
                {
                    _mm256_store_ps(mem, mask.m);
                }
                KSIMD_API(mask_t) test_load_mask(const scalar_t* mem) noexcept
                {
                    return { _mm256_load_ps(mem) };
                }
#endif

                // --- Comparison Operations ---

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

                KSIMD_API(mask_t) mask_and(mask_t lhs, mask_t rhs) noexcept
                {
                    return { _mm256_and_ps(lhs.m, rhs.m) };
                }

                KSIMD_API(mask_t) mask_or(mask_t lhs, mask_t rhs) noexcept
                {
                    return { _mm256_or_ps(lhs.m, rhs.m) };
                }

                KSIMD_API(mask_t) mask_xor(mask_t lhs, mask_t rhs) noexcept
                {
                    return { _mm256_xor_ps(lhs.m, rhs.m) };
                }

                KSIMD_API(mask_t) mask_not(mask_t mask) noexcept
                {
                    __m256 m = _mm256_set1_ps(OneBlock<scalar_t>);
                    return { _mm256_xor_ps(mask.m, m) };
                }

                KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
                {
                    return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q) };
                }

                KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
                {
                    return { _mm256_and_ps(_mm256_cmp_ps(lhs.v, lhs.v, _CMP_UNORD_Q),
                                           _mm256_cmp_ps(rhs.v, rhs.v, _CMP_UNORD_Q)) };
                }

                KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
                {
                    return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_ORD_Q) };
                }

                KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
                {
                    __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<scalar_t>);
                    __m256 inf_v = _mm256_set1_ps(Inf<scalar_t>);
                    return { _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                                          _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
                }

                KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
                {
                    __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<scalar_t>);
                    __m256 inf_v = _mm256_set1_ps(Inf<scalar_t>);

                    return { _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                                           _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
                }

                KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
                {
                    return { _mm256_blendv_ps(b.v, a.v, mask.m) };
                }

                KSIMD_API(scalar_t) reduce_add(batch_t v) noexcept
                {
                    __m128 low = _mm256_castps256_ps128(v.v);
                    __m128 high = _mm256_extractf128_ps(v.v, 0b1);
                    __m128 sum = _mm_add_ps(low, high);

                    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
                    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
                    return _mm_cvtss_f32(sum);
                }
            };
        } // namespace detail

        template<is_scalar_type S>
        using op = detail::op_impl<S>;
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace ksimd
#undef KSIMD_API

#include "operators.inl"

// do not use include guard

#include "kSimd/IDE/IDE_hint.hpp"

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1.1
#include <immintrin.h> // AVX+

#include <cstring> // memcpy

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/traits.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
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

    // --- constants ---
    template<is_scalar_type float32>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Lanes = 32 / sizeof(float32);

    template<is_scalar_type float32>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = 32;

    // --- any types ---
    KSIMD_API(Batch<float32>) load(const float32* mem) noexcept
    {
        return { _mm256_load_ps(mem) };
    }

    KSIMD_API(Batch<float32>) loadu(const float32* mem) noexcept
    {
        return { _mm256_loadu_ps(mem) };
    }

    KSIMD_API(Batch<float32>) load_partial(const float32* mem, size_t count) noexcept
    {
        count = count > Lanes<float32> ? Lanes<float32> : count;

        if (count == 0) [[unlikely]]
            return { _mm256_setzero_ps() };

        Batch<float32> res = { _mm256_setzero_ps() };
        std::memcpy(&res.v, mem, sizeof(float32) * count);
        return res;
    }

    KSIMD_API(void) store(float32* mem, Batch<float32> v) noexcept
    {
        _mm256_store_ps(mem, v.v);
    }

    KSIMD_API(void) storeu(float32* mem, Batch<float32> v) noexcept
    {
        _mm256_storeu_ps(mem, v.v);
    }

    KSIMD_API(void) store_partial(float32* mem, Batch<float32> v, size_t count) noexcept
    {
        count = count > Lanes<float32> ? Lanes<float32> : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v.v, sizeof(float32) * count);
    }

    template<std::same_as<float32> S>
    KSIMD_API(Batch<S>) undefined() noexcept
    {
        return { _mm256_undefined_ps() };
    }

    template<std::same_as<float32> S>
    KSIMD_API(Batch<S>) zero() noexcept
    {
        return { _mm256_setzero_ps() };
    }

    KSIMD_API(Batch<float32>) set(float32 x) noexcept
    {
        return { _mm256_set1_ps(x) };
    }

    template<std::same_as<float32> S>
    KSIMD_API(Batch<S>) sequence() noexcept
    {
        return { _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f) };
    }

    template<std::same_as<float32> S>
    KSIMD_API(Batch<S>) sequence(float32 base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return { _mm256_add_ps(iota, base_v) };
    }

    template<std::same_as<float32> S>
    KSIMD_API(Batch<S>) sequence(float32 base, float32 stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return { _mm256_fmadd_ps(stride_v, iota, base_v) };
    }

    KSIMD_API(Batch<float32>) add(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_add_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) sub(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_sub_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) mul(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_mul_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) div(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_div_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) mul_add(Batch<float32> a, Batch<float32> b, Batch<float32> c) noexcept
    {
        return { _mm256_fmadd_ps(a.v, b.v, c.v) };
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native>
    KSIMD_API(Batch<float32>) min(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q);
            __m256 min_v = _mm256_min_ps(lhs.v, rhs.v);
            __m256 nan_v = _mm256_set1_ps(QNaN<float32>);
            return { _mm256_blendv_ps(min_v, nan_v, has_nan) };
        }
        else
        {
            return { _mm256_min_ps(lhs.v, rhs.v) };
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native>
    KSIMD_API(Batch<float32>) max(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q);
            __m256 max_v = _mm256_max_ps(lhs.v, rhs.v);
            __m256 nan_v = _mm256_set1_ps(QNaN<float32>);
            return { _mm256_blendv_ps(max_v, nan_v, has_nan) };
        }
        else
        {
            return { _mm256_max_ps(lhs.v, rhs.v) };
        }
    }

    KSIMD_API(Batch<float32>) bit_not(Batch<float32> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(OneBlock<float32>);
        return { _mm256_xor_ps(v.v, mask) };
    }

    KSIMD_API(Batch<float32>) bit_and(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_and_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) bit_and_not(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_andnot_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) bit_or(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_or_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) bit_xor(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.v, rhs.v) };
    }

    KSIMD_API(Batch<float32>) bit_if_then_else(Batch<float32> _if, Batch<float32> _then, Batch<float32> _else) noexcept
    {
        return { _mm256_or_ps(_mm256_and_ps(_if.v, _then.v), _mm256_andnot_ps(_if.v, _else.v)) };
    }

#if defined(KSIMD_IS_TESTING)
    KSIMD_API(void) test_store_mask(float32* mem, Mask<float32> mask) noexcept
    {
        _mm256_store_ps(mem, mask.m);
    }
    KSIMD_API(Mask<float32>) test_load_mask(const float32* mem) noexcept
    {
        return { _mm256_load_ps(mem) };
    }
#endif

    // --- Comparison Operations ---

    KSIMD_API(Mask<float32>) equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    KSIMD_API(Mask<float32>) not_equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    KSIMD_API(Mask<float32>) greater(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    KSIMD_API(Mask<float32>) greater_equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    KSIMD_API(Mask<float32>) less(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    KSIMD_API(Mask<float32>) less_equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    KSIMD_API(Mask<float32>) mask_and(Mask<float32> lhs, Mask<float32> rhs) noexcept
    {
        return { _mm256_and_ps(lhs.m, rhs.m) };
    }

    KSIMD_API(Mask<float32>) mask_or(Mask<float32> lhs, Mask<float32> rhs) noexcept
    {
        return { _mm256_or_ps(lhs.m, rhs.m) };
    }

    KSIMD_API(Mask<float32>) mask_xor(Mask<float32> lhs, Mask<float32> rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.m, rhs.m) };
    }

    KSIMD_API(Mask<float32>) mask_not(Mask<float32> mask) noexcept
    {
        __m256 m = _mm256_set1_ps(OneBlock<float32>);
        return { _mm256_xor_ps(mask.m, m) };
    }

    KSIMD_API(Batch<float32>) if_then_else(Mask<float32> _if, Batch<float32> _then, Batch<float32> _else) noexcept
    {
        return { _mm256_blendv_ps(_else.v, _then.v, _if.m) };
    }

    KSIMD_API(float32) reduce_add(Batch<float32> v) noexcept
    {
        // [1, 2, 3, 4]
        __m128 low = _mm256_castps256_ps128(v.v);

        // [5, 6, 7, 8]
        __m128 high = _mm256_extractf128_ps(v.v, 0b1);

        // [15, 26, 37, 48]
        __m128 sum = _mm_add_ps(low, high);

        // [37, 48, 37, 48]
        __m128 shuffle1 = _mm_movehl_ps(sum, sum);

        // [1357, 2468, ...]
        sum = _mm_add_ps(sum, shuffle1);

        // [2468, ...]
        __m128 shuffle2 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1));

        // [12345678, ...]
        sum = _mm_add_ss(sum, shuffle2);

        return _mm_cvtss_f32(sum);
    }

    KSIMD_API(float32) reduce_mul(Batch<float32> v) noexcept
    {
        // [1, 2, 3, 4]
        __m128 low = _mm256_castps256_ps128(v.v);
        // [5, 6, 7, 8]
        __m128 high = _mm256_extractf128_ps(v.v, 0b1);

        // [15, 26, 37, 48]
        __m128 mul1 = _mm_mul_ps(low, high);

        // [37, 48, ...]
        __m128 shuffle1 = _mm_movehl_ps(mul1, mul1);

        // [1357, 2468, ...]
        __m128 mul2 = _mm_mul_ps(mul1, shuffle1);

        // [2468, ...]
        __m128 shuffle2 = _mm_shuffle_ps(mul2, mul2, _MM_SHUFFLE(1, 1, 1, 1));

        // [12345678, ...]
        __m128 res = _mm_mul_ss(mul2, shuffle2);

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native>
    KSIMD_API(float32) reduce_min(Batch<float32> v) noexcept
    {
        // [1, 2, 3, 4]
        // [5, 6, 7, 8]
        __m128 low1 = _mm256_castps256_ps128(v.v);
        __m128 high1 = _mm256_extractf128_ps(v.v, 0b1);

        // [ min(1,5), min(2,6), min(3,7), min(4,8) ]
        __m128 min1 = _mm_min_ps(low1, high1);

        // [ min(3,7), min(4,8), min(3,7), min(4,8) ]
        __m128 shuffle1 = _mm_movehl_ps(min1, min1);

        // [ min(1,3,5,7), min(2,4,6,8), ... ]
        __m128 min2 = _mm_min_ps(min1, shuffle1);

        // [ min(2,4,6,8), min(2,4,6,8), ... ]
        __m128 shuffle2 = _mm_shuffle_ps(min2, min2, _MM_SHUFFLE(1, 1, 1, 1));

        // [ min(1,2,3,4,5,6,7,8), ... ]
        __m128 res = _mm_min_ps(min2, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 nan_check = _mm256_cmp_ps(v.v, v.v, _CMP_UNORD_Q);
            int32 has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<float32> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native>
    KSIMD_API(float32) reduce_max(Batch<float32> v) noexcept
    {
        // [1, 2, 3, 4]
        // [5, 6, 7, 8]
        __m128 low1 = _mm256_castps256_ps128(v.v);
        __m128 high1 = _mm256_extractf128_ps(v.v, 0b1);

        // [ max(1,5), max(2,6), max(3,7), max(4,8) ]
        __m128 max1 = _mm_max_ps(low1, high1);

        // [ max(3,7), max(4,8), max(3,7), max(4,8) ]
        __m128 shuffle1 = _mm_movehl_ps(max1, max1);

        // [ max(1,3,5,7), max(2,4,6,8), ... ]
        __m128 max2 = _mm_max_ps(max1, shuffle1);

        // [ max(2,4,6,8), max(2,4,6,8), ... ]
        __m128 shuffle2 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(1, 1, 1, 1));

        // [ max(1,2,3,4,5,6,7,8), ... ]
        __m128 res = _mm_max_ps(max2, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 nan_check = _mm256_cmp_ps(v.v, v.v, _CMP_UNORD_Q);
            int32 has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<float32> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }

    // --- signed ---
    KSIMD_API(Batch<float32>) abs(Batch<float32> v) noexcept
    {
        return { _mm256_and_ps(v.v, _mm256_set1_ps(SignBitClearMask<float32>)) };
    }

    KSIMD_API(Batch<float32>) neg(Batch<float32> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(SignBitMask<float32>);
        return { _mm256_xor_ps(v.v, mask) };
    }

    // --- floating point ---
    KSIMD_API(Batch<float32>) sqrt(Batch<float32> v) noexcept
    {
        return { _mm256_sqrt_ps(v.v) };
    }

    template<RoundingMode mode>
    KSIMD_API(Batch<float32>) round(Batch<float32> v) noexcept
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
            __m256 sign_bit = _mm256_set1_ps(SignBitMask<float32>);
            __m256 half = _mm256_set1_ps(0x1.0p-1f);
            return { _mm256_round_ps(_mm256_add_ps(v.v, _mm256_or_ps(half, _mm256_and_ps(v.v, sign_bit))),
                                     _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
    }

    KSIMD_API(Mask<float32>) not_greater(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    KSIMD_API(Mask<float32>) not_greater_equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    KSIMD_API(Mask<float32>) not_less(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    KSIMD_API(Mask<float32>) not_less_equal(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    KSIMD_API(Mask<float32>) any_NaN(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    KSIMD_API(Mask<float32>) all_NaN(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_and_ps(_mm256_cmp_ps(lhs.v, lhs.v, _CMP_UNORD_Q), _mm256_cmp_ps(rhs.v, rhs.v, _CMP_UNORD_Q)) };
    }

    KSIMD_API(Mask<float32>) not_NaN(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_ORD_Q) };
    }

    KSIMD_API(Mask<float32>) any_finite(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
        __m256 inf_v = _mm256_set1_ps(Inf<float32>);
        return { _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                              _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
    }

    KSIMD_API(Mask<float32>) all_finite(Batch<float32> lhs, Batch<float32> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<float32>);
        __m256 inf_v = _mm256_set1_ps(Inf<float32>);

        return { _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                               _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
    }

    // --- float32 only ---
    KSIMD_API(Batch<float32>) rcp(Batch<float32> v) noexcept
    {
        return { _mm256_rcp_ps(v.v) };
    }

    KSIMD_API(Batch<float32>) rsqrt(Batch<float32> v) noexcept
    {
        return { _mm256_rsqrt_ps(v.v) };
    }
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

#include "operators.inl"

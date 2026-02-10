// do not use include guard

// #include "kSimd/IDE/IDE_hint.hpp"

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> // AVX+

#include <cstring> // memcpy

#include <bit> // bit_cast

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- types ---
    template<is_scalar_type>
    struct Batch
    {
        __m256i v;
    };

    template<>
    struct Batch<float>
    {
        __m256 v;
    };

#if KSIMD_SUPPORT_STD_FLOAT32
    template<>
    struct Batch<std::float32_t>
    {
        __m256 v;
    };
#endif

    template<>
    struct Batch<double>
    {
        __m256d v;
    };
    
#if KSIMD_SUPPORT_STD_FLOAT64
    template<>
    struct Batch<std::float64_t>
    {
        __m256d v;
    };
#endif

    template<is_scalar_type>
    struct Mask
    {
        __m256i m;
    };

    template<>
    struct Mask<float>
    {
        __m256 m;
    };
    
#if KSIMD_SUPPORT_STD_FLOAT32
    template<>
    struct Mask<std::float32_t>
    {
        __m256 m;
    };
#endif

    template<>
    struct Mask<double>
    {
        __m256d m;
    };
    
#if KSIMD_SUPPORT_STD_FLOAT64
    template<>
    struct Mask<std::float64_t>
    {
        __m256d m;
    };
#endif
#pragma endregion

#pragma region--- constants ---
    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Lanes = vec_size::Vec256 / sizeof(S);

    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Vec256;
#pragma endregion

#pragma region--- any types ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) load(const S* mem) noexcept
    {
        return { _mm256_load_ps(reinterpret_cast<const float*>(mem)) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) store(S* mem, Batch<S> v) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), v.v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu(const S* mem) noexcept
    {
        return { _mm256_loadu_ps(reinterpret_cast<const float*>(mem)) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu(S* mem, Batch<S> v) noexcept
    {
        _mm256_storeu_ps(reinterpret_cast<float*>(mem), v.v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu_partial(const S* mem, size_t count) noexcept
    {
        count = count > Lanes<S> ? Lanes<S> : count;

        Batch<S> res = { _mm256_setzero_ps() };

        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res.v, mem, sizeof(S) * count);
        return res;
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu_partial(S* mem, Batch<S> v, size_t count) noexcept
    {
        count = count > Lanes<S> ? Lanes<S> : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v.v, sizeof(S) * count);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) undefined() noexcept
    {
        return { _mm256_undefined_ps() };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) zero() noexcept
    {
        return { _mm256_setzero_ps() };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) set(S x) noexcept
    {
        return { _mm256_set1_ps(x) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence() noexcept
    {
        return { _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(S base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return { _mm256_add_ps(iota, base_v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(S base, S stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return { _mm256_fmadd_ps(stride_v, iota, base_v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) add(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_add_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sub(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_sub_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_mul_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) div(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_div_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul_add(Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        return { _mm256_fmadd_ps(a.v, b.v, c.v) };
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) min(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q);
            __m256 min_v = _mm256_min_ps(lhs.v, rhs.v);
            __m256 nan_v = _mm256_set1_ps(QNaN<S>);
            return { _mm256_blendv_ps(min_v, nan_v, has_nan) };
        }
        else
        {
            return { _mm256_min_ps(lhs.v, rhs.v) };
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) max(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q);
            __m256 max_v = _mm256_max_ps(lhs.v, rhs.v);
            __m256 nan_v = _mm256_set1_ps(QNaN<S>);
            return { _mm256_blendv_ps(max_v, nan_v, has_nan) };
        }
        else
        {
            return { _mm256_max_ps(lhs.v, rhs.v) };
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_not(Batch<S> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(OneBlock<S>);
        return { _mm256_xor_ps(v.v, mask) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_and_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and_not(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_andnot_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_or(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_or_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_xor(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.v, rhs.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_if_then_else(Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return { _mm256_or_ps(_mm256_and_ps(_if.v, _then.v), _mm256_andnot_ps(_if.v, _else.v)) };
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) test_store_mask(S* mem, Mask<S> mask) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), mask.m);
    }
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) test_load_mask(const S* mem) noexcept
    {
        return { _mm256_load_ps(reinterpret_cast<const float*>(mem)) };
    }
#endif

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_EQ_OQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NEQ_UQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GT_OQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_GE_OQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LT_OQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_LE_OQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_and(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { _mm256_and_ps(lhs.m, rhs.m) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_or(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { _mm256_or_ps(lhs.m, rhs.m) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_xor(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { _mm256_xor_ps(lhs.m, rhs.m) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_not(Mask<S> mask) noexcept
    {
        __m256 m = _mm256_set1_ps(OneBlock<S>);
        return { _mm256_xor_ps(mask.m, m) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) if_then_else(Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return { _mm256_blendv_ps(_else.v, _then.v, _if.m) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_add(Batch<S> v) noexcept
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

    template<is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_mul(Batch<S> v) noexcept
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_min(Batch<S> v) noexcept
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
            int32_t has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<S> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_max(Batch<S> v) noexcept
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
            int32_t has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<S> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }
#pragma endregion

#pragma region--- signed ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) abs(Batch<S> v) noexcept
    {
        return { _mm256_and_ps(v.v, _mm256_set1_ps(SignBitClearMask<S>)) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) neg(Batch<S> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(SignBitMask<S>);
        return { _mm256_xor_ps(v.v, mask) };
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sqrt(Batch<S> v) noexcept
    {
        return { _mm256_sqrt_ps(v.v) };
    }

    template<RoundingMode mode, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) round(Batch<S> v) noexcept
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
            __m256 sign_bit = _mm256_set1_ps(SignBitMask<S>);
            __m256 half = _mm256_set1_ps(0x1.0p-1f);
            return { _mm256_round_ps(_mm256_add_ps(v.v, _mm256_or_ps(half, _mm256_and_ps(v.v, sign_bit))),
                                     _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC) };
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGT_UQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NGE_UQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLT_UQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_NLE_UQ) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_UNORD_Q) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_and_ps(_mm256_cmp_ps(lhs.v, lhs.v, _CMP_UNORD_Q), _mm256_cmp_ps(rhs.v, rhs.v, _CMP_UNORD_Q)) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { _mm256_cmp_ps(lhs.v, rhs.v, _CMP_ORD_Q) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<S>);
        __m256 inf_v = _mm256_set1_ps(Inf<S>);
        return { _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                              _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<S>);
        __m256 inf_v = _mm256_set1_ps(Inf<S>);

        return { _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs.v, abs_mask), inf_v, _CMP_LT_OQ),
                               _mm256_cmp_ps(_mm256_and_ps(rhs.v, abs_mask), inf_v, _CMP_LT_OQ)) };
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Batch<S> v) noexcept
    {
        return { _mm256_rcp_ps(v.v) };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Batch<S> v) noexcept
    {
        return { _mm256_rsqrt_ps(v.v) };
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

#include "operators.inl"

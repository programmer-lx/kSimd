// do not use include guard

// #include "kSimd/IDE/IDE_hint.hpp"

#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> // AVX+

#include <cstring> // memcpy

#include "op_helpers.hpp"
#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- traits ---
    template<is_scalar_type S>
    struct Traits
    {
        using _scalar_type = S;
        static constexpr size_t _lanes = vec_size::Vec256 / sizeof(S);
    };

    template<is_scalar_type S>
    constexpr size_t lanes(Traits<S>) noexcept
    {
        return Traits<S>::_lanes;
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Vec256;
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<is_scalar_type>
        struct batch_type
        {
            using type = __m256i;
        };

        template<>
        struct batch_type<float>
        {
            using type = __m256;
        };

#if KSIMD_SUPPORT_STD_FLOAT32
        template<>
        struct batch_type<std::float32_t>
        {
            using type = __m256;
        };
#endif

        template<>
        struct batch_type<double>
        {
            using type = __m256d;
        };

#if KSIMD_SUPPORT_STD_FLOAT64
        template<>
        struct batch_type<std::float64_t>
        {
            using type = __m256d;
        };
#endif
    } // namespace detail

    template<is_scalar_type S>
    using Batch = detail::batch_type<S>::type;


    namespace detail
    {
        template<is_scalar_type>
        struct mask_type
        {
            using type = __m256i;
        };

        template<>
        struct mask_type<float>
        {
            using type = __m256;
        };

#if KSIMD_SUPPORT_STD_FLOAT32
        template<>
        struct mask_type<std::float32_t>
        {
            using type = __m256;
        };
#endif

        template<>
        struct mask_type<double>
        {
            using type = __m256d;
        };

#if KSIMD_SUPPORT_STD_FLOAT64
        template<>
        struct mask_type<std::float64_t>
        {
            using type = __m256d;
        };
#endif
    } // namespace detail

    template<is_scalar_type S>
    using Mask = detail::mask_type<S>::type;
#pragma endregion

#pragma region--- any types ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) load(Traits<S>, const S* mem) noexcept
    {
        return _mm256_load_ps(reinterpret_cast<const float*>(mem));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) store(Traits<S>, S* mem, Batch<S> v) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu(Traits<S>, const S* mem) noexcept
    {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(mem));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu(Traits<S>, S* mem, Batch<S> v) noexcept
    {
        _mm256_storeu_ps(reinterpret_cast<float*>(mem), v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu_partial(Traits<S>, const S* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Traits<S>{});
        count = count > L ? L : count;

        __m256 res = _mm256_setzero_ps();

        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(S) * count);
        return res;
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu_partial(Traits<S>, S* mem, Batch<S> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Traits<S>{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(S) * count);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) undefined(Traits<S>) noexcept
    {
        return _mm256_undefined_ps();
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) zero(Traits<S>) noexcept
    {
        return _mm256_setzero_ps();
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) set(Traits<S>, S x) noexcept
    {
        return _mm256_set1_ps(x);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S>) noexcept
    {
        return _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S>, S base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_add_ps(iota, base_v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S>, S base, S stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_fmadd_ps(stride_v, iota, base_v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) add(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_add_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sub(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_sub_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_mul_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul_add(Traits<S>, Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        return _mm256_fmadd_ps(a, b, c);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) min(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
            __m256 min_v = _mm256_min_ps(lhs, rhs);
            __m256 nan_v = _mm256_set1_ps(QNaN<S>);
            return _mm256_blendv_ps(min_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_min_ps(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) max(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
            __m256 max_v = _mm256_max_ps(lhs, rhs);
            __m256 nan_v = _mm256_set1_ps(QNaN<S>);
            return _mm256_blendv_ps(max_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_max_ps(lhs, rhs);
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_not(Traits<S>, Batch<S> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(OneBlock<S>);
        return _mm256_xor_ps(v, mask);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and_not(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_or(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_xor(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_if_then_else(Traits<S>, Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return _mm256_or_ps(_mm256_and_ps(_if, _then), _mm256_andnot_ps(_if, _else));
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) test_store_mask(Traits<S>, S* mem, Mask<S> mask) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), mask);
    }
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) test_load_mask(Traits<S>, const S* mem) noexcept
    {
        return _mm256_load_ps(reinterpret_cast<const float*>(mem));
    }
#endif

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_UQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_and(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_or(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_xor(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_not(Traits<S>, Mask<S> mask) noexcept
    {
        __m256 m = _mm256_set1_ps(OneBlock<S>);
        return _mm256_xor_ps(mask, m);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) if_then_else(Traits<S>, Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return _mm256_blendv_ps(_else, _then, _if);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_add(Traits<S>, Batch<S> v) noexcept
    {
        // [1, 2, 3, 4]
        __m128 low = _mm256_castps256_ps128(v);

        // [5, 6, 7, 8]
        __m128 high = _mm256_extractf128_ps(v, 0b1);

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
    KSIMD_API(S) reduce_mul(Traits<S>, Batch<S> v) noexcept
    {
        // [1, 2, 3, 4]
        __m128 low = _mm256_castps256_ps128(v);
        // [5, 6, 7, 8]
        __m128 high = _mm256_extractf128_ps(v, 0b1);

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
    KSIMD_API(S) reduce_min(Traits<S>, Batch<S> v) noexcept
    {
        // [1, 2, 3, 4]
        // [5, 6, 7, 8]
        __m128 low1 = _mm256_castps256_ps128(v);
        __m128 high1 = _mm256_extractf128_ps(v, 0b1);

        // [ min(1,5), min(2,6), min(3,7), min(4,8) ]
        __m128 min1 = _mm_min_ps(low1, high1);

        // [ min(3,7), min(4,8), min(3,7), min(4,8) ]
        __m128 shuffle1 = _mm_movehl_ps(min1, min1);

        // [ min(1,3,5,7), min(2,4,6,8), ... ]
        __m128 min2 = _mm_min_ps(min1, shuffle1);

        // [ min(2,4,6,8), min(2,4,6,8), ... ]
        __m128 shuffle2 = _mm_shuffle_ps(min2, min2, _MM_SHUFFLE(1, 1, 1, 1));

        // [ min(1,2,3,4,5,6,7,8), ... ]
        __m128 res = _mm_min_ss(min2, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 nan_check = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            int32_t has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<S> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_max(Traits<S>, Batch<S> v) noexcept
    {
        // [1, 2, 3, 4]
        // [5, 6, 7, 8]
        __m128 low1 = _mm256_castps256_ps128(v);
        __m128 high1 = _mm256_extractf128_ps(v, 0b1);

        // [ max(1,5), max(2,6), max(3,7), max(4,8) ]
        __m128 max1 = _mm_max_ps(low1, high1);

        // [ max(3,7), max(4,8), max(3,7), max(4,8) ]
        __m128 shuffle1 = _mm_movehl_ps(max1, max1);

        // [ max(1,3,5,7), max(2,4,6,8), ... ]
        __m128 max2 = _mm_max_ps(max1, shuffle1);

        // [ max(2,4,6,8), max(2,4,6,8), ... ]
        __m128 shuffle2 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(1, 1, 1, 1));

        // [ max(1,2,3,4,5,6,7,8), ... ]
        __m128 res = _mm_max_ss(max2, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 nan_check = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
            int32_t has_nan = _mm256_movemask_ps(nan_check);
            return has_nan ? QNaN<S> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }
#pragma endregion

#pragma region--- signed ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) abs(Traits<S>, Batch<S> v) noexcept
    {
        return _mm256_and_ps(v, _mm256_set1_ps(SignBitClearMask<S>));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) neg(Traits<S>, Batch<S> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(SignBitMask<S>);
        return _mm256_xor_ps(v, mask);
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) div(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_div_ps(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sqrt(Traits<S>, Batch<S> v) noexcept
    {
        return _mm256_sqrt_ps(v);
    }

    template<RoundingMode mode, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) round(Traits<S>, Batch<S> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return _mm256_round_ps(v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return _mm256_round_ps(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else /* if constexpr (mode == RoundingMode::Round) */
        {
            __m256 sign_bit = _mm256_set1_ps(SignBitMask<S>);

            __m256 half = _mm256_set1_ps(0.5f);

            // 构造一个与v具有相同符号的0.5
            __m256 half_with_sign_bit = _mm256_or_ps(half, _mm256_and_ps(v, sign_bit));

            return _mm256_round_ps(_mm256_add_ps(v, half_with_sign_bit), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGT_UQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGE_UQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLT_UQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLE_UQ);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_and_ps(_mm256_cmp_ps(lhs, lhs, _CMP_UNORD_Q), _mm256_cmp_ps(rhs, rhs, _CMP_UNORD_Q));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_ORD_Q);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<S>);
        __m256 inf_v = _mm256_set1_ps(Inf<S>);
        return _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                            _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<S>);
        __m256 inf_v = _mm256_set1_ps(Inf<S>);

        return _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                             _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Traits<S>, Batch<S> v) noexcept
    {
        return _mm256_rcp_ps(v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Traits<S>, Batch<S> v) noexcept
    {
        return _mm256_rsqrt_ps(v);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

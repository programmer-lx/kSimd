// do not use include guard

#include <immintrin.h> // AVX+

#include <cstring>

#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

// 复用SSE的逻辑，实现 Fixed128Tag
#include "x86_sse_family.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- constants ---
    template<is_tag_256 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        return vec_size::Vec256 / sizeof(tag_scalar_t<Tag>);
    }
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<typename Tag, typename Enable>
        struct batch_type;

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = __m256;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = __m256d;
        };
    } // namespace detail

    namespace detail
    {
        template<typename Tag, typename Enable>
        struct mask_type;

        // f32
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = __m256;
        };

        // f64
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = __m256d;
        };
    } // namespace detail

    // public user types
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion

#pragma region--- any types ---
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_load_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_storeu_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;

        __m256 res = _mm256_setzero_ps();

        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm256_undefined_ps();
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm256_setzero_ps();
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm256_set1_ps(x);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_add_ps(iota, base_v);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_fmadd_ps(stride_v, iota, base_v);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mul_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return _mm256_fmadd_ps(a, b, c);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
            __m256 min_v = _mm256_min_ps(lhs, rhs);
            __m256 nan_v = _mm256_set1_ps(QNaN<tag_scalar_t<Tag>>);
            return _mm256_blendv_ps(min_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_min_ps(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256 has_nan = _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
            __m256 max_v = _mm256_max_ps(lhs, rhs);
            __m256 nan_v = _mm256_set1_ps(QNaN<tag_scalar_t<Tag>>);
            return _mm256_blendv_ps(max_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_max_ps(lhs, rhs);
        }
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(v, mask);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm256_or_ps(_mm256_and_ps(_if, _then), _mm256_andnot_ps(_if, _else));
    }

#if defined(KSIMD_IS_TESTING)
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) test_store_mask(Tag, tag_scalar_t<Tag>* mem, Mask<Tag> mask) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), mask);
    }
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) test_load_mask(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_load_ps(reinterpret_cast<const float*>(mem));
    }
#endif

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_UQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        __m256 m = _mm256_set1_ps(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(mask, m);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm256_blendv_ps(_else, _then, _if);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
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

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
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
            return has_nan ? QNaN<tag_scalar_t<Tag>> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
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
            return has_nan ? QNaN<tag_scalar_t<Tag>> : _mm_cvtss_f32(res);
        }

        return _mm_cvtss_f32(res);
    }
#pragma endregion

#pragma region--- signed ---
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_and_ps(v, _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>));
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(SignBitMask<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(v, mask);
    }
#pragma endregion

#pragma region--- floating point ---
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_div_ps(lhs, rhs);
    }

    template<typename Tag> requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sqrt_ps(v);
    }

    template<RoundingMode mode, typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
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
            __m256 sign_bit = _mm256_set1_ps(SignBitMask<tag_scalar_t<Tag>>);

            __m256 half = _mm256_set1_ps(0.5f);

            // 构造一个与v具有相同符号的0.5
            __m256 half_with_sign_bit = _mm256_or_ps(half, _mm256_and_ps(v, sign_bit));

            return _mm256_round_ps(_mm256_add_ps(v, half_with_sign_bit), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGT_UQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGE_UQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLT_UQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLE_UQ);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_ps(_mm256_cmp_ps(lhs, lhs, _CMP_UNORD_Q), _mm256_cmp_ps(rhs, rhs, _CMP_UNORD_Q));
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_ORD_Q);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256 inf_v = _mm256_set1_ps(Inf<tag_scalar_t<Tag>>);
        return _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                            _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256 inf_v = _mm256_set1_ps(Inf<tag_scalar_t<Tag>>);

        return _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                             _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }
#pragma endregion

#pragma region--- float32 only ---
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_rcp_ps(v);
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_rsqrt_ps(v);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

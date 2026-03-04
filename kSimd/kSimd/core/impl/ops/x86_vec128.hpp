// do not use include guard

#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_SSE_END
    #include <xmmintrin.h> // SSE
    #include <emmintrin.h> // SSE2
    #include <pmmintrin.h> // SSE3
    #include <tmmintrin.h> // SSSE3
    #include <smmintrin.h> // SSE4.1

    #include <cstring> // memcpy
#endif

#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_AVX512_END
    #include <immintrin.h> // AVX+
#endif

#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_SSE_END
    #include "shared.hpp"
#endif

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

#if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
    #define KSIMD_IS_TAG_F32_OR_FAKE_F16(tag) is_tag_float_16bits<tag> // native f16 (AVX512-FP16)
#else
    #define KSIMD_IS_TAG_F32_OR_FAKE_F16(tag) (is_tag_float_32bits<tag> || is_tag_float_16bits<tag>)
#endif

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_128 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        return vec_size::Vec128 / sizeof(tag_scalar_t<Tag>);
    }
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<typename Tag, typename Enable>
        struct batch_type;

        // native f16 (AVX512-FP16)
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = __m128h;
        };
        #endif

        // f32 or f16(fake)
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag)>>
        {
            using type = __m128;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = __m128d;
        };

        template<typename Tag, typename Enable>
        struct mask_type;

// avx512
#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        // 32bits
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_scalar_32<Tag>>>
        {
            using type = __mmask8;
        };

        // 64bits
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_scalar_64<Tag>>>
        {
            using type = __mmask8;
        };
// sse
#elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        // mask 跟 batch 一样
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag>>>
        {
            using type = typename detail::batch_type<Tag, void>::type;
        };
#endif

        template<typename Tag, typename Enable>
        struct mask_bitset_type;

        template<typename Tag>
        struct mask_bitset_type<Tag, std::enable_if_t<is_tag_128<Tag>>>
        {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
            using type = typename detail::mask_type<Tag, void>::type;
        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
            using type = int; // 指令集返回的是int
        #endif
        };
    } // namespace detail

    // public user types
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;

    template<is_tag Tag>
    using MaskBitset = typename detail::mask_bitset_type<Tag, void>::type;
#pragma endregion


#pragma region--- impl ---
    namespace detail
    {
        // f16: only lower half is valid
        KSIMD_API(__m128) mm_f16_to_f32(__m128i f16) noexcept
        {
            // f16 = [?, ?, ?, ?, d, c, b, a]

            // 把 f16 的前半部分的每个元素向高位位移 16bit，占满整个 __m128i
            // [0, d, 0, c, 0, b, 0, a]
            __m128i v32 = _mm_unpacklo_epi16(f16, _mm_setzero_si128());

            // [d, 0, c, 0, b, 0, a, 0]
            __m128i w = _mm_slli_epi32(v32, 16);

            __m128i sign = _mm_and_si128(w, _mm_set1_epi32(SignBitMask<int32_t>));

            __m128i two_w = _mm_add_epi32(w, w);

            __m128i exp_offset = _mm_set1_epi32(INT32_C(0xE0) << 23);
            __m128 exp_scale = _mm_set1_ps(std::bit_cast<float>(INT32_C(0x7800000)));
            __m128 normalized_value = _mm_castsi128_ps(_mm_add_epi32(_mm_srli_epi32(two_w, 4), exp_offset));
            normalized_value = _mm_mul_ps(normalized_value, exp_scale);

            __m128i magic_mask = _mm_set1_epi32(INT32_C(126) << 23);
            __m128 magic_bias = _mm_set1_ps(0.5f);
            __m128 denormalized_value = _mm_castsi128_ps(_mm_or_si128(_mm_srli_epi32(two_w, 17), magic_mask));
            denormalized_value = _mm_sub_ps(denormalized_value, magic_bias);

            __m128i denormalized_cutoff = _mm_set1_epi32(INT32_C(1) << 27);

            // cond 相当于 !(two_w < denormalized_cutoff)，因为SSE没有无符号比较指令，只能这样子实现无符号的比较
            __m128i max_val = _mm_max_epu32(two_w, denormalized_cutoff);
            __m128i cond = _mm_cmpeq_epi32(max_val, two_w);

            __m128i denormalized_value_bits = _mm_castps_si128(denormalized_value);
            __m128i normalized_value_bits = _mm_castps_si128(normalized_value);

            __m128i result = _mm_blendv_epi8(denormalized_value_bits, normalized_value_bits, cond);
            result = _mm_or_si128(result, sign);

            return _mm_castsi128_ps(result);
        }

        // return value: only lower half is valid
        KSIMD_API(__m128i) mm_f32_to_f16(__m128 f32) noexcept
        {
            __m128 scale_to_inf = _mm_set1_ps(std::bit_cast<float>(UINT32_C(0x77800000)));
            __m128 scale_to_zero = _mm_set1_ps(std::bit_cast<float>(UINT32_C(0x08800000)));
            __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float>);
            __m128 saturated_f = _mm_mul_ps(_mm_and_ps(f32, abs_mask), scale_to_inf);

            __m128 base = _mm_mul_ps(saturated_f, scale_to_zero);

            __m128i w = _mm_castps_si128(f32);
            __m128i shl1_w = _mm_add_epi32(w, w);
            __m128i sign = _mm_and_si128(w, _mm_set1_epi32(SignBitMask<int32_t>));

            // 1111 1111 0000 0000 ....
            __m128i bias_mask = _mm_set1_epi32(0xff << 24);
            __m128i bias = _mm_and_si128(shl1_w, bias_mask);

            // if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
            bias = _mm_max_epu32(_mm_set1_epi32(INT32_C(0x71000000)), bias);

            __m128i magic = _mm_add_epi32(_mm_srli_epi32(bias, 1), _mm_set1_epi32(INT32_C(0x07800000)));
            base = _mm_add_ps(_mm_castsi128_ps(magic), base);
            __m128i bits = _mm_castps_si128(base);
            __m128i exp_bits = _mm_and_si128(_mm_srli_epi32(bits, 13), _mm_set1_epi32(INT32_C(0x00007C00)));
            __m128i mantissa_bits = _mm_and_si128(bits, _mm_set1_epi32(INT32_C(0x00000FFF)));
            __m128i nosign = _mm_add_epi32(exp_bits, mantissa_bits);

            // cond = shl1_w > UINT32(0xFF000000)
            // sse没有无符号比较指令，而0xFF000000有符号位，所以必须采用以下的指令来修正
            // if shl1_w <= 0xFF000000: min_val = shl1_w        -> cond = true
            // if shl1_w > 0xFF000000: min_val = 0xFF000000     -> cond = false
            // num = shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign
            // 等价于 cond ? nonsign : UINT16_C(0x7E00)
            __m128i min_val = _mm_min_epu32(shl1_w, _mm_set1_epi32(INT32_C(0xff) << 24));
            __m128i cond = _mm_cmpeq_epi32(shl1_w, min_val);
            __m128i f16 = _mm_blendv_epi8(_mm_set1_epi32(0x7E00), nosign, cond);
            __m128i sign_f16 = _mm_srli_epi32(sign, 16);
            f16 = _mm_or_si128(sign_f16, f16);

            // 现在的 f16 是 [?, d, ?, c, ?, b, ?, a]
            // 现在要将他变成 [?, ?, ?, ?, d, c, b, a]，将数据移动到低64bit
            __m128i result = _mm_packus_epi32(f16, f16);

            return result;
        }
    }
#pragma endregion


#pragma region--- any types ---

#pragma region--- any type/float32 ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm_load_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm_store_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm_loadu_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm_storeu_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        __mmask8 mask = _mm_cmp_ps_mask(iota, cnt, _CMP_LT_OQ);
        return _mm_maskz_loadu_ps(mask, mem);

        // avx
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        __m128 mask = _mm_cmp_ps(iota, cnt, _CMP_LT_OQ);
        return _mm_maskload_ps(reinterpret_cast<const float*>(mem), _mm_castps_si128(mask));

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;

        __m128 res = _mm_setzero_ps();

        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(float) * count);
        return res;
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        __mmask8 mask = _mm_cmp_ps_mask(iota, cnt, _CMP_LT_OQ);
        _mm_mask_storeu_ps(mem, mask, v);

        // avx
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        __m128 mask = _mm_cmp_ps(iota, cnt, _CMP_LT_OQ);
        return _mm_maskstore_ps(reinterpret_cast<float*>(mem), _mm_castps_si128(mask), v);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(float) * count);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm_undefined_ps();
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm_setzero_ps();
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, float x) noexcept
    {
        return _mm_set1_ps(x);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm_set_ps(3.f, 2.f, 1.f, 0.f);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, float base) noexcept
    {
        __m128 base_v = _mm_set1_ps(base);
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        return _mm_add_ps(iota, base_v);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, float base, float stride) noexcept
    {
        __m128 stride_v = _mm_set1_ps(stride);
        __m128 base_v = _mm_set1_ps(base);
        __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);

        // avx v3 v4
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        return _mm_fmadd_ps(stride_v, iota, base_v);

        // sse family
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_add_ps(_mm_mul_ps(stride_v, iota), base_v);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm_add_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm_sub_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm_mul_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // avx v3 v4
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        return _mm_fmadd_ps(a, b, c);

        // sse family
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_add_ps(_mm_mul_ps(a, b), c);
        #endif
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // avx512
            #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
            __mmask8 has_nan = _mm_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);
            __m128 min_v = _mm_min_ps(lhs, rhs);
            __m128 nan_v = _mm_set1_ps(QNaN<float>);
            return _mm_mask_blend_ps(has_nan, min_v, nan_v);

            // sse
            #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
            __m128 has_nan = _mm_cmpunord_ps(lhs, rhs);
            __m128 min_v = _mm_min_ps(lhs, rhs);
            __m128 nan_v = _mm_set1_ps(QNaN<float>);
            return _mm_blendv_ps(min_v, nan_v, has_nan);
            #endif
        }
        else
        {
            return _mm_min_ps(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // avx512
            #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
            __mmask8 has_nan = _mm_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);
            __m128 max_v = _mm_max_ps(lhs, rhs);
            __m128 nan_v = _mm_set1_ps(QNaN<float>);
            return _mm_mask_blend_ps(has_nan, max_v, nan_v);

            // sse
            #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
            __m128 has_nan = _mm_cmpunord_ps(lhs, rhs);
            __m128 max_v = _mm_max_ps(lhs, rhs);
            __m128 nan_v = _mm_set1_ps(QNaN<float>);
            return _mm_blendv_ps(max_v, nan_v, has_nan);
            #endif
        }
        else
        {
            return _mm_max_ps(lhs, rhs);
        }
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        __m128 mask = _mm_set1_ps(OneBlock<float>);
        return _mm_xor_ps(v, mask);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        return _mm_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        return _mm_andnot_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        return _mm_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        return _mm_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        KSIMD_DETAIL_UNSUPPORT_FP16_BIT_OP(Tag);
        
        return _mm_or_ps(_mm_and_ps(_if, _then), _mm_andnot_ps(_if, _else));
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpeq_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_NEQ_UQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpneq_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_GT_OQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpgt_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_GE_OQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpge_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmplt_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmple_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _kand_mask8(lhs, rhs);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_and_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _kor_mask8(lhs, rhs);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_or_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _kxor_mask8(lhs, rhs);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_xor_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _knot_mask8(mask);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        __m128 m = _mm_set1_ps(OneBlock<float>);
        return _mm_xor_ps(mask, m);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _kandn_mask8(lhs, rhs);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_andnot_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_mask_blend_ps(_if, _else, _then);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_blendv_ps(_else, _then, _if);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        static_assert(!is_tag_float_16bits<Tag>, "TODO");

        // [2, 1, 4, 3]
        __m128 shuffle1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));

        // [12, 12, 34, 34]
        __m128 sum1 = _mm_add_ps(v, shuffle1);

        // [34, ...]
        __m128 shuffle2 = _mm_movehl_ps(sum1, sum1);

        // [1234, ...]
        __m128 sum2 = _mm_add_ss(sum1, shuffle2);

        return _mm_cvtss_f32(sum2);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        static_assert(!is_tag_float_16bits<Tag>, "TODO");

        // [2, 1, 4, 3]
        __m128 shuffle1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));

        // [12, 12, 34, 34]
        __m128 mul1 = _mm_mul_ps(v, shuffle1);

        // [34, ...]
        __m128 shuffle2 = _mm_movehl_ps(mul1, mul1);

        // [1234, ...]
        __m128 mul2 = _mm_mul_ss(mul1, shuffle2);

        return _mm_cvtss_f32(mul2);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        static_assert(!is_tag_float_16bits<Tag>, "TODO");

        // [2, 1, 4, 3]
        __m128 shuffle1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));

        // [ min(1,2), min(1,2), min(3,4), min(3,4) ]
        __m128 min1 = _mm_min_ps(v, shuffle1);

        // [ min(3,4), ... ]
        __m128 shuffle2 = _mm_movehl_ps(min1, min1);

        // [ min(1,2,3,4), ... ]
        __m128 res = _mm_min_ss(min1, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // avx512
            #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
            __mmask8 nan_check = _mm_cmp_ps_mask(v, v, _CMP_UNORD_Q);
            unsigned char no_nan = _ktestz_mask8_u8(nan_check, nan_check);
            return no_nan ? _mm_cvtss_f32(res) : QNaN<float>;

            // sse
            #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
            __m128 nan_check = _mm_cmpunord_ps(v, v);
            int32_t has_nan = _mm_movemask_ps(nan_check);
            return has_nan ? QNaN<float> : _mm_cvtss_f32(res);
            #endif
        }

        return _mm_cvtss_f32(res);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        static_assert(!is_tag_float_16bits<Tag>, "TODO");

        // [2, 1, 4, 3]
        __m128 shuffle1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));

        // [ max(1,2), max(1,2), max(3,4), max(3,4) ]
        __m128 max1 = _mm_max_ps(v, shuffle1);

        // [ max(3,4), ... ]
        __m128 shuffle2 = _mm_movehl_ps(max1, max1);

        // [ max(1,2,3,4), ... ]
        __m128 res = _mm_max_ss(max1, shuffle2);

        // NaN传播
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // avx512
            #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
            __mmask8 nan_check = _mm_cmp_ps_mask(v, v, _CMP_UNORD_Q);
            unsigned char no_nan = _ktestz_mask8_u8(nan_check, nan_check);
            return no_nan ? _mm_cvtss_f32(res) : QNaN<float>;

            // sse
            #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
            __m128 nan_check = _mm_cmpunord_ps(v, v);
            int32_t has_nan = _mm_movemask_ps(nan_check);
            return has_nan ? QNaN<float> : _mm_cvtss_f32(res);
            #endif
        }

        return _mm_cvtss_f32(res);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(MaskBitset<Tag>) reduce_mask(Tag, Mask<Tag> mask) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return mask;

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_movemask_ps(mask);
        #endif
    }
#pragma endregion // any type/float32

#pragma region--- any type/float16 ---
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        #error TODO native fp16: load

        // f16c
        #elif KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V3
        __m128i f16 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mem));
        return _mm_cvtph_ps(f16);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        __m128i f16 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mem));
        return detail::mm_f16_to_f32(f16);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        #error TODO native fp16: load

        // f16c
        #elif KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V3
        __m128i f16 = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(mem), f16);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        __m128i f16 = detail::mm_f32_to_f16(v);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(mem), f16);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        
        #else
        return load(t, mem);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        
        #else
        storeu(t, mem, v);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* , size_t ) noexcept
    {
        // TODO
        return undefined(Tag{});

        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        // avx512
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        // __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        // __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        // __mmask8 mask = _mm_cmp_ps_mask(iota, cnt, _CMP_LT_OQ);
        // return _mm_maskz_loadu_ps(mask, mem);

        // avx
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        // __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        // __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        // __m128 mask = _mm_cmp_ps(iota, cnt, _CMP_LT_OQ);
        // return _mm_maskload_ps(reinterpret_cast<const float*>(mem), _mm_castps_si128(mask));

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        // constexpr size_t L = lanes(Tag{});
        // count = count > L ? L : count;
        //
        // __m128 res = _mm_setzero_ps();
        //
        // if (count == 0) [[unlikely]]
        //     return res;
        //
        // std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        // return res;
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* , Batch<Tag> , size_t ) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        // avx512
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        // __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        // __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        // __mmask8 mask = _mm_cmp_ps_mask(iota, cnt, _CMP_LT_OQ);
        // _mm_mask_storeu_ps(mem, mask, v);

        // avx
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START
        // __m128 iota = _mm_set_ps(3.f, 2.f, 1.f, 0.f);
        // __m128 cnt = _mm_set1_ps(static_cast<float>(count));
        // __m128 mask = _mm_cmp_ps(iota, cnt, _CMP_LT_OQ);
        // return _mm_maskstore_ps(reinterpret_cast<float*>(mem), _mm_castps_si128(mask), v);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        // constexpr size_t L = lanes(Tag{});
        // count = count > L ? L : count;
        // if (count == 0) [[unlikely]]
        //     return;
        //
        // std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
        #endif
    }
#pragma endregion

#pragma endregion // any type

#pragma region--- signed ---
    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm_and_ps(v, _mm_set1_ps(SignBitClearMask<float>));
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        __m128 mask = _mm_set1_ps(SignBitMask<float>);
        return _mm_xor_ps(v, mask);
    }
#pragma endregion // signed

#pragma region--- floating point ---
    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm_div_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm_sqrt_ps(v);
    }

    template<RoundingMode mode, typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return _mm_round_ps(v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return _mm_round_ps(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return _mm_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else /* if constexpr (mode == RoundingMode::Round) */
        {
            __m128 sign_bit = _mm_set1_ps(SignBitMask<float>);

            __m128 half = _mm_set1_ps(0.5f);

            // 构造一个与v具有相同符号的0.5
            __m128 half_with_sign_bit = _mm_or_ps(half, _mm_and_ps(v, sign_bit));

            return _mm_round_ps(_mm_add_ps(v, half_with_sign_bit), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_NGT_UQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpngt_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_NGE_UQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpnge_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_NLT_UQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpnlt_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_NLE_UQ);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpnle_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpunord_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _kand_mask8(
            _mm_cmp_ps_mask(lhs, lhs, _CMP_UNORD_Q),
            _mm_cmp_ps_mask(rhs, rhs, _CMP_UNORD_Q));

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_and_ps(_mm_cmpunord_ps(lhs, lhs), _mm_cmpunord_ps(rhs, rhs));
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        return _mm_cmp_ps_mask(lhs, rhs, _CMP_ORD_Q);

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        return _mm_cmpord_ps(lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float>);
        __m128 inf_v = _mm_set1_ps(Inf<float>);
        return _kor_mask8(_mm_cmp_ps_mask(_mm_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
            _mm_cmp_ps_mask(_mm_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float>);
        __m128 inf_v = _mm_set1_ps(Inf<float>);
        return _mm_or_ps(_mm_cmplt_ps(_mm_and_ps(lhs, abs_mask), inf_v),
                         _mm_cmplt_ps(_mm_and_ps(rhs, abs_mask), inf_v));
        #endif
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512
        #if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX512_START
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float>);
        __m128 inf_v = _mm_set1_ps(Inf<float>);

        return _kand_mask8(_mm_cmp_ps_mask(_mm_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
            _mm_cmp_ps_mask(_mm_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));

        // sse
        #elif KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START
        __m128 abs_mask = _mm_set1_ps(SignBitClearMask<float>);
        __m128 inf_v = _mm_set1_ps(Inf<float>);

        return _mm_and_ps(_mm_cmplt_ps(_mm_and_ps(lhs, abs_mask), inf_v),
                          _mm_cmplt_ps(_mm_and_ps(rhs, abs_mask), inf_v));
        #endif
    }
#pragma endregion // floating point

#pragma region--- float32 only ---
    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return _mm_rcp_ps(v);
    }

    template<typename Tag>
        requires(KSIMD_IS_TAG_F32_OR_FAKE_F16(Tag) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm_rsqrt_ps(v);
    }
#pragma endregion // float32 only

#pragma region--- cast ---
    template<typename Tag_To, typename Tag_From>
        requires(is_tag_128<Tag_To> && is_tag_128<Tag_From>)
    KSIMD_API(Batch<Tag_To>) demote(Tag_To, Tag_From, Batch<Tag_From> v) noexcept
    {
        static_assert(sizeof(tag_scalar_t<Tag_To>) < sizeof(tag_scalar_t<Tag_From>),
            "sizeof(To) must less than sizeof(From).");

        // f16 <- f32
        if constexpr (is_tag_float_16bits<Tag_To> && is_tag_float_32bits<Tag_From>)
        {
            // do nothing
            return v;
        }
        else
        {
            static_assert(!is_tag_128<Tag_To>, "unreachable.");
            return zero(Tag_To{});
        }
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_IS_TAG_F32_OR_FAKE_F16
#undef KSIMD_API

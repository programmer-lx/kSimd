// do not use include guard

#include <immintrin.h> // AVX+

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

// 复用SSE的逻辑，实现 Fixed128Tag
#include "x86_vec128.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- constants ---
    template<is_tag_512 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32 x 16)
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        if constexpr (is_tag_float_16bits<Tag>)
        {
            return vec_size::Vec512 / sizeof(float);
        }
        #endif
        return vec_size::Vec512 / sizeof(tag_scalar_t<Tag>);
    }
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        // batch
        template<typename Tag, typename Enable>
        struct batch_type;

        // fake f16
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = __m512;
        };
        // AVX512-FP16
        #else
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = __m512h;
        };
        #endif

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = __m512;
        };

        // int32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_int32<Tag>>>
        {
            using type = __m512i;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = __m512d;
        };

        // mask
        template<typename Tag, typename Enable>
        struct mask_type;

        // 16bits (需要对 fake 16 进行特殊处理)
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_scalar_16<Tag>>>
        {
            #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
            using type = __mmask16;
            #else
            using type = __mmask32;
            #endif
        };

        // 32bits
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_scalar_32<Tag>>>
        {
            using type = __mmask16;
        };

        // 64bits
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_512<Tag> && is_tag_scalar_64<Tag>>>
        {
            using type = __mmask8;
        };
    } // namespace detail

    // public user types
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion

#pragma region--- impl ---
#pragma endregion

#pragma region--- any types ---
#pragma region--- any types/load ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm512_load_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm512_load_si512(mem);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __m256i f16 = _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
        return _mm512_cvtph_ps(f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/load

#pragma region--- any types/store ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm512_store_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm512_store_si512(reinterpret_cast<void*>(mem), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __m256i f16 = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm256_store_si256(reinterpret_cast<__m256i*>(mem), f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/store

#pragma region--- any types/loadu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm512_loadu_si512(mem);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __m256i f16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
        return _mm512_cvtph_ps(f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/loadu

#pragma region--- any types/storeu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm512_storeu_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm512_storeu_si512(reinterpret_cast<void*>(mem), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __m256i f16 = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/storeu

#pragma region--- any types/loadu_partial ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        __mmask16 mask = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));

        return _mm512_maskz_loadu_ps(mask, mem);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        __mmask16 m = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));
        return _mm512_maskz_loadu_epi32(m, mem);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __mmask16 mask = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));

        __m256i f16 = _mm256_maskz_loadu_epi16(mask, mem);
        return _mm512_cvtph_ps(f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/loadu_partial

#pragma region--- any types/storeu_partial ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        __mmask16 mask = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));

        _mm512_mask_storeu_ps(mem, mask, v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        __mmask16 m = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));
        _mm512_mask_storeu_epi32(mem, m, v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        __mmask16 mask = static_cast<__mmask16>((UINT16_C(1) << count) - UINT16_C(1));

        __m256i f16 = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm256_mask_storeu_epi16(mem, mask, f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/storeu_partial

#pragma region--- any types/undefined ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm512_undefined_ps();
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm512_setzero_si512();
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return undefined(detail::Tag512<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/undefined

#pragma region--- any types/zero ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm512_setzero_ps();
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm512_setzero_si512();
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return zero(detail::Tag512<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/zero

#pragma region--- any types/set ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm512_set1_ps(x);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm512_set1_epi32(x);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return set(detail::Tag512<float>{}, static_cast<float>(x));
        #endif
    }
#endif // FP16
#pragma endregion // any types/set

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm512_set_ps(
            15.f, 14.f, 13.f, 12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag512<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        __m512 base_v = _mm512_set1_ps(base);
        __m512 iota = _mm512_set_ps(
            15.f, 14.f, 13.f, 12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm512_add_ps(iota, base_v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm512_add_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                                _mm512_set1_epi32(base));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag512<float>{}, static_cast<float>(base));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m512 stride_v = _mm512_set1_ps(stride);
        __m512 base_v = _mm512_set1_ps(base);
        __m512 iota = _mm512_set_ps(
            15.f, 14.f, 13.f, 12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm512_fmadd_ps(stride_v, iota, base_v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        __m512i base_v = _mm512_set1_epi32(base);
        __m512i stride_v = _mm512_set1_epi32(stride);
        return _mm512_add_epi32(_mm512_mullo_epi32(iota, stride_v), base_v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag512<float>{}, static_cast<float>(base), static_cast<float>(stride));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_add_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_add_epi32(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return add(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/add

#pragma region--- any types/sub ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_sub_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_sub_epi32(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sub(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/sub

#pragma region--- any types/mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_mul_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_mullo_epi32(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mul(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul

#pragma region--- any types/mul_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return _mm512_fmadd_ps(a, b, c);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mul_add(detail::Tag512<float>{}, a, b, c);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul_add

#pragma region--- any types/min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __mmask16 has_nan = _mm512_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);
            __m512 min_v = _mm512_min_ps(lhs, rhs);
            __m512 nan_v = _mm512_set1_ps(QNaN<tag_scalar_t<Tag>>);
            return _mm512_mask_blend_ps(has_nan, min_v, nan_v);
        }
        else
        {
            return _mm512_min_ps(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_min_epi32(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return min<option>(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/min

#pragma region--- any types/max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __mmask16 has_nan = _mm512_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);
            __m512 max_v = _mm512_max_ps(lhs, rhs);
            __m512 nan_v = _mm512_set1_ps(QNaN<tag_scalar_t<Tag>>);
            return _mm512_mask_blend_ps(has_nan, max_v, nan_v);
        }
        else
        {
            return _mm512_max_ps(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_max_epi32(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return max<option>(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/max

#pragma region--- any types/bit_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        __m512 mask = _mm512_set1_ps(OneBlock<tag_scalar_t<Tag>>);
        return _mm512_xor_ps(v, mask);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept { return _mm512_xor_si512(v, _mm512_set1_epi32(-1)); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_not

#pragma region--- any types/bit_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_and_si512(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and

#pragma region--- any types/bit_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_andnot_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_andnot_si512(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and_not

#pragma region--- any types/bit_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_or_si512(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_or

#pragma region--- any types/bit_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return _mm512_xor_si512(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_xor

#pragma region--- any types/equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_EQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/equal

#pragma region--- any types/not_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NEQ_UQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_NE);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/not_equal

#pragma region--- any types/greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_GT_OQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_GT);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return greater(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater

#pragma region--- any types/greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_GE_OQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_GE);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return greater_equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater_equal

#pragma region--- any types/less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_LT);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return less(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less

#pragma region--- any types/less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_epi32_mask(lhs, rhs, _MM_CMPINT_LE);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return less_equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less_equal

#pragma region--- any types/mask_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _kand_mask16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept { return _kand_mask16(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_and(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and

#pragma region--- any types/mask_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _kor_mask16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept { return _kor_mask16(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_or(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_or

#pragma region--- any types/mask_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _kxor_mask16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept { return _kxor_mask16(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_xor(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_xor

#pragma region--- any types/mask_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _knot_mask16(mask);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept { return _knot_mask16(mask); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_not(detail::Tag512<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_not

#pragma region--- any types/mask_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _kandn_mask16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept { return _kandn_mask16(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_and_not(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and_not

#pragma region--- any types/mask_all ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == lane_mask;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == lane_mask;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == lane_mask;
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_all

#pragma region--- any types/mask_any ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) != 0;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) != 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) != 0;
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_any

#pragma region--- any types/mask_none ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == 0;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        constexpr int lane_mask = 0b1111'1111'1111'1111;
        return (mask & lane_mask) == 0;
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_none

#pragma region--- any types/if_then_else ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm512_mask_blend_ps(_if, _else, _then);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm512_mask_blend_epi32(_if, _else, _then);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return if_then_else(detail::Tag512<float>{}, _if, _then, _else);
        #endif
    }
#endif // FP16
#pragma endregion // any types/if_then_else

#pragma region--- any types/reduce_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_add_ps(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_add_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_add(detail::Tag512<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_add

#pragma region--- any types/reduce_mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_mul_ps(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_mul_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_mul(detail::Tag512<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_mul

#pragma region--- any types/reduce_min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            float res = _mm512_reduce_min_ps(v);
            __mmask16 nan_check = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
            unsigned char no_nan = _kortestz_mask16_u8(nan_check, nan_check);
            return no_nan ? res : QNaN<tag_scalar_t<Tag>>;
        }
        else
        {
            return _mm512_reduce_min_ps(v);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_min_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_min<option>(detail::Tag512<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_min

#pragma region--- any types/reduce_max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            float res = _mm512_reduce_max_ps(v);
            __mmask16 nan_check = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);
            unsigned char no_nan = _kortestz_mask16_u8(nan_check, nan_check);
            return no_nan ? res : QNaN<tag_scalar_t<Tag>>;
        }
        else
        {
            return _mm512_reduce_max_ps(v);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_reduce_max_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_max<option>(detail::Tag512<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_max

#pragma endregion // any types

#pragma region--- signed ---
#pragma region--- signed/abs ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_abs_ps(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_abs_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        #else
        return abs(detail::Tag512<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // signed/abs

#pragma region--- signed/neg ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        __m512 mask = _mm512_set1_ps(SignBitMask<tag_scalar_t<Tag>>);
        return _mm512_xor_ps(v, mask);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_sub_epi32(_mm512_setzero_si512(), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        #else
        return neg(detail::Tag512<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // signed/neg

#pragma endregion // signed

#pragma region--- floating point ---

#pragma region--- floating point/div ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_div_ps(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return div(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/div

#pragma region--- floating point/sqrt ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_sqrt_ps(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sqrt(detail::Tag512<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/sqrt

#pragma region--- floating point/round ---
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return _mm512_roundscale_ps(v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return _mm512_roundscale_ps(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else /* if constexpr (mode == RoundingMode::Round) */
        {
            __m512 sign_bit = _mm512_set1_ps(SignBitMask<tag_scalar_t<Tag>>);

            __m512 half = _mm512_set1_ps(0.5f);

            // 构造一个与v具有相同符号的0.5
            __m512 half_with_sign_bit = _mm512_or_ps(half, _mm512_and_ps(v, sign_bit));

            return _mm512_roundscale_ps(_mm512_add_ps(v, half_with_sign_bit), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return round<mode>(detail::Tag512<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/round

#pragma region--- floating point/not_greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NGT_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_greater(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater

#pragma region--- floating point/not_greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NGE_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_greater_equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater_equal

#pragma region--- floating point/not_less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NLT_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_less(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less

#pragma region--- floating point/not_less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_NLE_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_less_equal(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less_equal

#pragma region--- floating point/any_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_UNORD_Q);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return any_NaN(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_NaN

#pragma region--- floating point/all_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _kand_mask16(_mm512_cmp_ps_mask(lhs, lhs, _CMP_UNORD_Q), _mm512_cmp_ps_mask(rhs, rhs, _CMP_UNORD_Q));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return all_NaN(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_NaN

#pragma region--- floating point/not_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm512_cmp_ps_mask(lhs, rhs, _CMP_ORD_Q);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_NaN(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_NaN

#pragma region--- floating point/any_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m512 inf_v = _mm512_set1_ps(Inf<tag_scalar_t<Tag>>);
        return _kor_mask16(_mm512_cmp_ps_mask(_mm512_abs_ps(lhs), inf_v, _CMP_LT_OQ),
            _mm512_cmp_ps_mask(_mm512_abs_ps(rhs), inf_v, _CMP_LT_OQ));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return any_finite(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_finite

#pragma region--- floating point/all_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m512 inf_v = _mm512_set1_ps(Inf<tag_scalar_t<Tag>>);

        return _kand_mask16(_mm512_cmp_ps_mask(_mm512_abs_ps(lhs), inf_v, _CMP_LT_OQ),
            _mm512_cmp_ps_mask(_mm512_abs_ps(rhs), inf_v, _CMP_LT_OQ));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return all_finite(detail::Tag512<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_finite

#pragma endregion // floating point

#pragma region--- float32 only ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_rcp14_ps(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_512<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm512_rsqrt14_ps(v);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

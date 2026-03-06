// do not use include guard

#include <immintrin.h> // AVX+
#include <cstring> // memcpy

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

// 复用SSE的逻辑，实现 Fixed128Tag
#include "x86_vec128.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- constants ---
    template<is_tag_256 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32 x 8)
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16
        if constexpr (is_tag_float_16bits<Tag>)
        {
            return vec_size::Vec256 / sizeof(float);
        }
        #endif
        return vec_size::Vec256 / sizeof(tag_scalar_t<Tag>);
    }
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        // batch
        template<typename Tag, typename Enable>
        struct batch_type;

        // fake f16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = __m256;
        };

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = __m256;
        };

        // int32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_int32<Tag>>>
        {
            using type = __m256i;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = __m256d;
        };

        // mask
        template<typename Tag, typename Enable>
        struct mask_type;

        // mask 跟 batch 一样
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_256<Tag>>>
        {
            using type = typename detail::batch_type<Tag, void>::type;
        };
    } // namespace detail

    // public user types
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion

#pragma region--- any types ---
#pragma region--- any types/load ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_load_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(mem));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        __m128i f16 = _mm_load_si128(reinterpret_cast<const __m128i*>(mem));
        return _mm256_cvtph_ps(f16);
    }
#endif // FP16
#pragma endregion // any types/load

#pragma region--- any types/store ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_store_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(mem), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        __m128i f16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_store_si128(reinterpret_cast<__m128i*>(mem), f16);
    }
#endif // FP16
#pragma endregion // any types/store

#pragma region--- any types/loadu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        __m128i f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mem));
        return _mm256_cvtph_ps(f16);
    }
#endif // FP16
#pragma endregion // any types/loadu

#pragma region--- any types/storeu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_storeu_ps(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mem), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        __m128i f16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(mem), f16);
    }
#endif // FP16
#pragma endregion // any types/storeu

#pragma region--- any types/loadu_partial ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;

        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        __m256 cnt = _mm256_set1_ps(static_cast<float>(count));
        __m256 mask = _mm256_cmp_ps(iota, cnt, _CMP_LT_OQ);

        if (count == 0) [[unlikely]]
            return _mm256_setzero_ps();

        return _mm256_maskload_ps(reinterpret_cast<const float*>(mem), _mm256_castps_si256(mask));
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_si256();

        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi32(static_cast<int32_t>(count));
        __m256i mask = _mm256_cmpgt_epi32(cnt, iota);
        return _mm256_maskload_epi32(mem, mask);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;

        if (count == 0) [[unlikely]]
            return _mm256_setzero_ps();

        __m128i f16 = _mm_setzero_si128();
        std::memcpy(&f16, mem, sizeof(tag_scalar_t<Tag>) * count);
        return _mm256_cvtph_ps(f16);
    }
#endif // FP16
#pragma endregion // any types/loadu_partial

#pragma region--- any types/storeu_partial ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        __m256 cnt = _mm256_set1_ps(static_cast<float>(count));
        __m256 mask = _mm256_cmp_ps(iota, cnt, _CMP_LT_OQ);

        _mm256_maskstore_ps(reinterpret_cast<float*>(mem), _mm256_castps_si256(mask), v);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;

        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi32(static_cast<int32_t>(count));
        __m256i mask = _mm256_cmpgt_epi32(cnt, iota);
        _mm256_maskstore_epi32(mem, mask, v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        __m128i f16 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        std::memcpy(mem, &f16, sizeof(tag_scalar_t<Tag>) * count);
    }
#endif // FP16
#pragma endregion // any types/storeu_partial

#pragma region--- any types/undefined ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm256_undefined_ps();
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm256_setzero_si256();
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return undefined(detail::Tag256<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/undefined

#pragma region--- any types/zero ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm256_setzero_ps();
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm256_setzero_si256();
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return zero(detail::Tag256<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/zero

#pragma region--- any types/set ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm256_set1_ps(x);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm256_set1_epi32(x);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return set(detail::Tag256<float>{}, static_cast<float>(x));
        #endif
    }
#endif // FP16
#pragma endregion // any types/set

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag256<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_add_ps(iota, base_v);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm256_add_epi32(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), _mm256_set1_epi32(base));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag256<float>{}, static_cast<float>(base));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256 stride_v = _mm256_set1_ps(stride);
        __m256 base_v = _mm256_set1_ps(base);
        __m256 iota = _mm256_set_ps(7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
        return _mm256_fmadd_ps(stride_v, iota, base_v);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i base_v = _mm256_set1_epi32(base);
        __m256i stride_v = _mm256_set1_epi32(stride);
        return _mm256_add_epi32(_mm256_mullo_epi32(iota, stride_v), base_v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sequence(detail::Tag256<float>{}, static_cast<float>(base), static_cast<float>(stride));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return add(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/add

#pragma region--- any types/sub ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sub(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/sub

#pragma region--- any types/mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mul_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mullo_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mul(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul

#pragma region--- any types/mul_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return _mm256_fmadd_ps(a, b, c);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return add(t, mul(t, a, b), c);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mul_add(detail::Tag256<float>{}, a, b, c);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul_add

#pragma region--- any types/min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return min<option>(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/min

#pragma region--- any types/max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return max<option>(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/max

#pragma region--- any types/bit_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(v, mask);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_xor_si256(v, _mm256_set1_epi32(-1));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag>) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_not

#pragma region--- any types/bit_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and

#pragma region--- any types/bit_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_andnot_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and_not

#pragma region--- any types/bit_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_or_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_or

#pragma region--- any types/bit_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_xor_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_xor

#pragma region--- any types/equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/equal

#pragma region--- any types/not_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NEQ_UQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, equal(t, lhs, rhs));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/not_equal

#pragma region--- any types/greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpgt_epi32(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return greater(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater

#pragma region--- any types/greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return greater_equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater_equal

#pragma region--- any types/less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return less(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less

#pragma region--- any types/less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return less_equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less_equal

#pragma region--- any types/mask_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_and_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_and_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_and(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and

#pragma region--- any types/mask_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_or_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_or_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_or(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_or

#pragma region--- any types/mask_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_xor_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_xor_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_xor(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_xor

#pragma region--- any types/mask_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        __m256 m = _mm256_set1_ps(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(mask, m);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _mm256_xor_si256(mask, _mm256_set1_epi32(-1));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_not(detail::Tag256<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_not

#pragma region--- any types/mask_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_andnot_ps(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_andnot_si256(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return mask_and_not(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and_not

#pragma region--- any types/mask_all ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_ps(mask);
        return m == 0b11111111; // 8 lanes are true
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        constexpr int32_t test = std::bit_cast<int32_t>(UINT32_C(0b1000'1000'1000'1000'1000'1000'1000'1000));
        return (m & test) == test;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        int m = _mm256_movemask_ps(mask);
        return m == 0b11111111; // 8 lanes are true
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_all

#pragma region--- any types/mask_any ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_ps(mask);
        return m != 0;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m != 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        int m = _mm256_movemask_ps(mask);
        return m != 0;
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_any

#pragma region--- any types/mask_none ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_ps(mask);
        return m == 0;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        int m = _mm256_movemask_ps(mask);
        return m == 0;
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_none

#pragma region--- any types/if_then_else ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm256_blendv_ps(_else, _then, _if);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm256_or_si256(_mm256_and_si256(_if, _then), _mm256_andnot_si256(_if, _else));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return if_then_else(detail::Tag256<float>{}, _if, _then, _else);
        #endif
    }
#endif // FP16
#pragma endregion // any types/if_then_else

#pragma region--- any types/reduce_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag t, Batch<Tag> v) noexcept
    {
        alignas(32) int32_t tmp[8];
        store(t, tmp, v);
        int32_t s = 0;
        for (size_t i = 0; i < 8; ++i) s += tmp[i];
        return s;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_add(detail::Tag256<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_add

#pragma region--- any types/reduce_mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag t, Batch<Tag> v) noexcept
    {
        alignas(32) int32_t tmp[8];
        store(t, tmp, v);
        int32_t p = 1;
        for (size_t i = 0; i < 8; ++i) p *= tmp[i];
        return p;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_mul(detail::Tag256<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_mul

#pragma region--- any types/reduce_min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag t, Batch<Tag> v) noexcept
    {
        alignas(32) int32_t tmp[8];
        store(t, tmp, v);
        int32_t m = tmp[0];
        for (size_t i = 1; i < 8; ++i) m = ksimd::min(m, tmp[i]);
        return m;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_min<option>(detail::Tag256<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_min

#pragma region--- any types/reduce_max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag t, Batch<Tag> v) noexcept
    {
        alignas(32) int32_t tmp[8];
        store(t, tmp, v);
        int32_t m = tmp[0];
        for (size_t i = 1; i < 8; ++i) m = ksimd::max(m, tmp[i]);
        return m;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_max<option>(detail::Tag256<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_max

#pragma endregion // any types

#pragma region--- signed ---
#pragma region--- signed/abs ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_and_ps(v, _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>));
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_abs_epi32(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return abs(detail::Tag256<float>{}, v);
    }
#endif
#pragma endregion // signed/abs

#pragma region--- signed/neg ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        __m256 mask = _mm256_set1_ps(SignBitMask<tag_scalar_t<Tag>>);
        return _mm256_xor_ps(v, mask);
    }

    template<typename Tag>
        requires(is_tag_256<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sub_epi32(_mm256_setzero_si256(), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return neg(detail::Tag256<float>{}, v);
    }
#endif
#pragma endregion // signed/neg

#pragma endregion // signed

#pragma region--- floating point ---
#pragma region--- floating point/div ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_div_ps(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return div(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }

#endif // FP16
#pragma endregion // floating point/div

#pragma region--- floating point/sqrt ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sqrt_ps(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return sqrt(detail::Tag256<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/sqrt

#pragma region--- floating point/round ---
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
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

#if KSIMD_SUPPORT_NATIVE_FP16
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return round<mode>(detail::Tag256<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/round

#pragma region--- floating point/not_greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGT_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_greater(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater

#pragma region--- floating point/not_greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NGE_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_greater_equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater_equal

#pragma region--- floating point/not_less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLT_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_less(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less

#pragma region--- floating point/not_less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_NLE_UQ);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_less_equal(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less_equal

#pragma region--- floating point/any_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_UNORD_Q);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return any_NaN(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_NaN

#pragma region--- floating point/all_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_ps(_mm256_cmp_ps(lhs, lhs, _CMP_UNORD_Q), _mm256_cmp_ps(rhs, rhs, _CMP_UNORD_Q));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return all_NaN(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_NaN

#pragma region--- floating point/not_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_ps(lhs, rhs, _CMP_ORD_Q);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return not_NaN(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_NaN

#pragma region--- floating point/any_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256 inf_v = _mm256_set1_ps(Inf<tag_scalar_t<Tag>>);
        return _mm256_or_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                            _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return any_finite(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_finite

#pragma region--- floating point/all_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256 abs_mask = _mm256_set1_ps(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256 inf_v = _mm256_set1_ps(Inf<tag_scalar_t<Tag>>);

        return _mm256_and_ps(_mm256_cmp_ps(_mm256_and_ps(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                             _mm256_cmp_ps(_mm256_and_ps(rhs, abs_mask), inf_v, _CMP_LT_OQ));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // avx512 fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_X86_V4_FULL_FP16

        #else
        return all_finite(detail::Tag256<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_finite

#pragma endregion // floating point

#pragma region--- float32 only ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_rcp_ps(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_rsqrt_ps(v);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION
#undef KSIMD_API

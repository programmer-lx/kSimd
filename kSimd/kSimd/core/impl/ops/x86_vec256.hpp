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

        // any integer
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_integer<Tag>>>
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

#pragma region--- impl ---
    KSIMD_API(__m256i) ksimd_mm256_mullo_epi8(__m256i lhs, __m256i rhs) noexcept
    {
        __m256i mask_lo = _mm256_set1_epi16(0x00FF);
        __m256i lo_a = _mm256_and_si256(lhs, mask_lo);
        __m256i lo_b = _mm256_and_si256(rhs, mask_lo);
        __m256i lo_mul = _mm256_and_si256(_mm256_mullo_epi16(lo_a, lo_b), mask_lo);
        __m256i hi_a = _mm256_srli_epi16(lhs, 8);
        __m256i hi_b = _mm256_srli_epi16(rhs, 8);
        __m256i hi_mul = _mm256_slli_epi16(_mm256_mullo_epi16(hi_a, hi_b), 8);
        return _mm256_or_si256(lo_mul, hi_mul);
    }

    KSIMD_API(__m256i) ksimd_mm256_mullo_epi64(__m256i lhs, __m256i rhs) noexcept
    {
        __m256i lo_lo = _mm256_mul_epu32(lhs, rhs);
        __m256i lhs_hi = _mm256_srli_epi64(lhs, 32);
        __m256i rhs_hi = _mm256_srli_epi64(rhs, 32);
        __m256i cross1 = _mm256_mul_epu32(lhs, rhs_hi);
        __m256i cross2 = _mm256_mul_epu32(lhs_hi, rhs);
        __m256i cross = _mm256_add_epi64(cross1, cross2);
        __m256i cross_shift = _mm256_slli_epi64(cross, 32);
        return _mm256_add_epi64(lo_lo, cross_shift);
    }
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_load_pd(reinterpret_cast<const double*>(mem));
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_store_pd(reinterpret_cast<double*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(mem));
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        _mm256_storeu_pd(reinterpret_cast<double*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_pd();

        __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d cnt = _mm256_set1_pd(static_cast<double>(count));
        __m256d mask = _mm256_cmp_pd(iota, cnt, _CMP_LT_OQ);
        return _mm256_maskload_pd(reinterpret_cast<const double*>(mem), _mm256_castpd_si256(mask));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_si256();
        __m256i res = _mm256_setzero_si256();
        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_si256();
        __m256i res = _mm256_setzero_si256();
        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_si256();
        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi32(static_cast<int32_t>(count));
        __m256i mask = _mm256_cmpgt_epi32(cnt, iota);
        return _mm256_maskload_epi32(reinterpret_cast<const int32_t*>(mem), mask);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return _mm256_setzero_si256();
        __m256i iota = _mm256_set_epi64x(3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi64x(static_cast<int64_t>(count));
        __m256i mask = _mm256_cmpgt_epi64(cnt, iota);
        return _mm256_maskload_epi64(reinterpret_cast<const long long*>(mem), mask);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;

        __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d cnt = _mm256_set1_pd(static_cast<double>(count));
        __m256d mask = _mm256_cmp_pd(iota, cnt, _CMP_LT_OQ);
        _mm256_maskstore_pd(reinterpret_cast<double*>(mem), _mm256_castpd_si256(mask), v);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;
        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;
        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;
        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi32(static_cast<int32_t>(count));
        __m256i mask = _mm256_cmpgt_epi32(cnt, iota);
        _mm256_maskstore_epi32(reinterpret_cast<int32_t*>(mem), mask, v);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(Tag{});
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;
        __m256i iota = _mm256_set_epi64x(3, 2, 1, 0);
        __m256i cnt = _mm256_set1_epi64x(static_cast<int64_t>(count));
        __m256i mask = _mm256_cmpgt_epi64(cnt, iota);
        _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem), mask, v);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return _mm256_undefined_pd();
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return _mm256_setzero_pd();
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm256_set1_pd(x);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return _mm256_set1_epi8(std::bit_cast<int8_t>(x));
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_int32<Tag> || is_tag_int64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        if constexpr (is_tag_int64<Tag>)
        {
            return _mm256_set1_epi64x(std::bit_cast<int64_t>(x));
        }
        else if constexpr (is_tag_int16<Tag>)
        {
            return _mm256_set1_epi16(std::bit_cast<int16_t>(x));
        }
        else
        {
            return _mm256_set1_epi32(std::bit_cast<int32_t>(x));
        }
    }

    template<typename Tag>
        requires((is_tag_uint16<Tag> || is_tag_uint32<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        if constexpr (is_tag_uint64<Tag>)
        {
            return _mm256_set1_epi64x(std::bit_cast<int64_t>(x));
        }
        else if constexpr (is_tag_uint16<Tag>)
        {
            return _mm256_set1_epi16(std::bit_cast<int16_t>(x));
        }
        else
        {
            return _mm256_set1_epi32(std::bit_cast<int32_t>(x));
        }
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_epi8(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return _mm256_set_epi64x(3, 2, 1, 0);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        __m256d base_v = _mm256_set1_pd(base);
        __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        return _mm256_add_pd(iota, base_v);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm256_add_epi8(
            _mm256_set_epi8(
                31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm256_set1_epi8(std::bit_cast<int8_t>(base))
        );
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm256_add_epi16(
            _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            _mm256_set1_epi16(std::bit_cast<int16_t>(base))
        );
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm256_add_epi32(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0), _mm256_set1_epi32(std::bit_cast<int32_t>(base)));
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return _mm256_add_epi64(_mm256_set_epi64x(3, 2, 1, 0), _mm256_set1_epi64x(std::bit_cast<int64_t>(base)));
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256d stride_v = _mm256_set1_pd(stride);
        __m256d base_v = _mm256_set1_pd(base);
        __m256d iota = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        return _mm256_fmadd_pd(stride_v, iota, base_v);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256i iota = _mm256_set_epi8(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        __m256i base_v = _mm256_set1_epi8(std::bit_cast<int8_t>(base));
        __m256i stride_v = _mm256_set1_epi8(std::bit_cast<int8_t>(stride));
        return _mm256_add_epi8(ksimd_mm256_mullo_epi8(iota, stride_v), base_v);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256i iota = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        __m256i base_v = _mm256_set1_epi16(std::bit_cast<int16_t>(base));
        __m256i stride_v = _mm256_set1_epi16(std::bit_cast<int16_t>(stride));
        return _mm256_add_epi16(_mm256_mullo_epi16(iota, stride_v), base_v);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        __m256i iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        __m256i base_v = _mm256_set1_epi32(std::bit_cast<int32_t>(base));
        __m256i stride_v = _mm256_set1_epi32(std::bit_cast<int32_t>(stride));
        return _mm256_add_epi32(_mm256_mullo_epi32(iota, stride_v), base_v);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        const uint64_t b = std::bit_cast<uint64_t>(base);
        const uint64_t s = std::bit_cast<uint64_t>(stride);
        return _mm256_set_epi64x(
            std::bit_cast<int64_t>(b + UINT64_C(3) * s),
            std::bit_cast<int64_t>(b + UINT64_C(2) * s),
            std::bit_cast<int64_t>(b + UINT64_C(1) * s),
            std::bit_cast<int64_t>(b)
        );
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_pd(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_epi8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_epi16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_epi32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_add_epi64(lhs, rhs);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_pd(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_epi8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_epi16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_epi32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_sub_epi64(lhs, rhs);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mul_pd(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return ksimd_mm256_mullo_epi8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mullo_epi16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_mullo_epi32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return ksimd_mm256_mullo_epi64(lhs, rhs);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return _mm256_fmadd_pd(a, b, c);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return add(t, mul(t, a, b), c);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return add(t, mul(t, a, b), c);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return add(t, mul(t, a, b), c);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256d has_nan = _mm256_cmp_pd(lhs, rhs, _CMP_UNORD_Q);
            __m256d min_v = _mm256_min_pd(lhs, rhs);
            __m256d nan_v = _mm256_set1_pd(QNaN<tag_scalar_t<Tag>>);
            return _mm256_blendv_pd(min_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_min_pd(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epi8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epu8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epi16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epu16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epi32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_min_epu32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i gt = _mm256_cmpgt_epi64(lhs, rhs);
        return _mm256_blendv_epi8(lhs, rhs, gt);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi64x(std::bit_cast<int64_t>(UINT64_C(0x8000000000000000)));
        __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
        return _mm256_blendv_epi8(lhs, rhs, gt);
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256d has_nan = _mm256_cmp_pd(lhs, rhs, _CMP_UNORD_Q);
            __m256d max_v = _mm256_max_pd(lhs, rhs);
            __m256d nan_v = _mm256_set1_pd(QNaN<tag_scalar_t<Tag>>);
            return _mm256_blendv_pd(max_v, nan_v, has_nan);
        }
        else
        {
            return _mm256_max_pd(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epi8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epu8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epi16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epu16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epi32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_max_epu32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i gt = _mm256_cmpgt_epi64(lhs, rhs);
        return _mm256_blendv_epi8(rhs, lhs, gt);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi64x(std::bit_cast<int64_t>(UINT64_C(0x8000000000000000)));
        __m256i gt = _mm256_cmpgt_epi64(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
        return _mm256_blendv_epi8(rhs, lhs, gt);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        __m256d mask = _mm256_set1_pd(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_pd(v, mask);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_andnot_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_or_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_EQ_OQ);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpeq_epi8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpeq_epi16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpeq_epi32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpeq_epi64(lhs, rhs);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NEQ_UQ);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_GT_OQ);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpgt_epi8(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80u));
        return _mm256_cmpgt_epi8(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpgt_epi16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi16(static_cast<int16_t>(0x8000));
        return _mm256_cmpgt_epi16(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpgt_epi32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi32(std::bit_cast<int32_t>(UINT32_C(0x80000000)));
        return _mm256_cmpgt_epi32(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmpgt_epi64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256i flip = _mm256_set1_epi64x(std::bit_cast<int64_t>(UINT64_C(0x8000000000000000)));
        return _mm256_cmpgt_epi64(_mm256_xor_si256(lhs, flip), _mm256_xor_si256(rhs, flip));
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_GE_OQ);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, greater(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return greater(t, rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_or(t, less(t, lhs, rhs), equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_and_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_or_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_xor_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        __m256d m = _mm256_set1_pd(OneBlock<tag_scalar_t<Tag>>);
        return _mm256_xor_pd(mask, m);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _mm256_xor_si256(mask, _mm256_set1_epi8(-1));
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _mm256_xor_si256(mask, _mm256_set1_epi16(-1));
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _mm256_xor_si256(mask, _mm256_set1_epi32(-1));
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return _mm256_xor_si256(mask, _mm256_set1_epi64x(-1));
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return _mm256_andnot_pd(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_pd(mask);
        return m == 0b1111;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == -1;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == -1;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        constexpr int32_t test = std::bit_cast<int32_t>(UINT32_C(0b1000'1000'1000'1000'1000'1000'1000'1000));
        return (m & test) == test;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        constexpr int32_t test = std::bit_cast<int32_t>(UINT32_C(0b10000000'10000000'10000000'10000000));
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_pd(mask);
        return m != 0;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m != 0;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m != 0;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m != 0;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        constexpr int32_t test = std::bit_cast<int32_t>(UINT32_C(0b10000000'10000000'10000000'10000000));
        return (m & test) != 0;
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_pd(mask);
        return m == 0;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == 0;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == 0;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        return m == 0;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        int m = _mm256_movemask_epi8(mask);
        constexpr int32_t test = std::bit_cast<int32_t>(UINT32_C(0b10000000'10000000'10000000'10000000));
        return (m & test) == 0;
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return _mm256_blendv_pd(_else, _then, _if);
    }

    template<typename Tag>
        requires(is_tag_integer<Tag> && is_tag_256<Tag>)
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d high = _mm256_extractf128_pd(v, 0b1);
        __m128d sum = _mm_add_pd(low, high);
        __m128d shuffle = _mm_shuffle_pd(sum, sum, 0b01);
        sum = _mm_add_sd(sum, shuffle);
        return _mm_cvtsd_f64(sum);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        __m256i sum256 = _mm256_sad_epu8(v, _mm256_setzero_si256());
        __m128i lo = _mm256_castsi256_si128(sum256);
        __m128i hi = _mm256_extracti128_si256(sum256, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        __m128i sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtsi128_si32(sum64);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) tag_scalar_t<Tag> lanes_v[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        int32_t acc = static_cast<int32_t>(lanes_v[0]);
        for (size_t i = 1; i < 16; ++i)
        {
            acc = acc + lanes_v[i];
        }
        return acc;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 0b1);
        __m128i sum = _mm_add_epi32(low, high);
        __m128i shuffle1 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(3, 2, 3, 2));
        sum = _mm_add_epi32(sum, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
        sum = _mm_add_epi32(sum, shuffle2);
        return std::bit_cast<tag_scalar_t<Tag>>(_mm_cvtsi128_si32(sum));
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        uint64_t sum = lanes_v[0] + lanes_v[1] + lanes_v[2] + lanes_v[3];
        return std::bit_cast<tag_scalar_t<Tag>>(sum);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d high = _mm256_extractf128_pd(v, 0b1);
        __m128d prod = _mm_mul_pd(low, high);
        __m128d shuffle = _mm_shuffle_pd(prod, prod, 0b01);
        prod = _mm_mul_sd(prod, shuffle);
        return _mm_cvtsd_f64(prod);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_256<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) tag_scalar_t<Tag> lanes_v[32];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        int32_t acc = lanes_v[0];
        for (size_t i = 1; i < 32; ++i)
        {
            acc = acc * lanes_v[i];
        }
        return acc;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_256<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) tag_scalar_t<Tag> lanes_v[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        int32_t acc = lanes_v[0];
        for (size_t i = 1; i < 16; ++i)
        {
            acc = acc * lanes_v[i];
        }
        return acc;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 0b1);
        __m128i mul1 = _mm_mullo_epi32(low, high);
        __m128i shuffle1 = _mm_shuffle_epi32(mul1, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i mul2 = _mm_mullo_epi32(mul1, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(mul2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128i res = _mm_mullo_epi32(mul2, shuffle2);
        return std::bit_cast<tag_scalar_t<Tag>>(_mm_cvtsi128_si32(res));
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        uint64_t prod = lanes_v[0] * lanes_v[1] * lanes_v[2] * lanes_v[3];
        return std::bit_cast<tag_scalar_t<Tag>>(prod);
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d high = _mm256_extractf128_pd(v, 0b1);
        __m128d min1 = _mm_min_pd(low, high);
        __m128d shuffle = _mm_shuffle_pd(min1, min1, 0b01);
        __m128d res = _mm_min_sd(min1, shuffle);

        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256d nan_check = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
            int32_t has_nan = _mm256_movemask_pd(nan_check);
            return has_nan ? QNaN<tag_scalar_t<Tag>> : _mm_cvtsd_f64(res);
        }
        return _mm_cvtsd_f64(res);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i min_vec = _mm_min_epi8(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        min_vec = _mm_min_epi8(min_vec, _mm_shuffle_epi32(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epi8(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epi8(min_vec, _mm_srli_epi32(min_vec, 16));
        min_vec = _mm_min_epi8(min_vec, _mm_srli_epi16(min_vec, 8));
        return static_cast<int8_t>(_mm_extract_epi8(min_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i min_vec = _mm_min_epu8(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        min_vec = _mm_min_epu8(min_vec, _mm_shuffle_epi32(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epu8(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epu8(min_vec, _mm_srli_epi32(min_vec, 16));
        min_vec = _mm_min_epu8(min_vec, _mm_srli_epi16(min_vec, 8));
        return static_cast<uint8_t>(_mm_extract_epi8(min_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i min_vec = _mm_min_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        min_vec = _mm_min_epi16(min_vec, _mm_shuffle_epi32(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epi16(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epi16(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 1, 1, 1)));
        return static_cast<uint16_t>(_mm_extract_epi16(min_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i min_vec = _mm_min_epu16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        min_vec = _mm_min_epu16(min_vec, _mm_shuffle_epi32(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epu16(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        min_vec = _mm_min_epu16(min_vec, _mm_shufflelo_epi16(min_vec, _MM_SHUFFLE(1, 1, 1, 1)));
        return static_cast<uint16_t>(_mm_extract_epi16(min_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i low1 = _mm256_castsi256_si128(v);
        __m128i high1 = _mm256_extracti128_si256(v, 0b1);
        __m128i min1 = _mm_min_epi32(low1, high1);
        __m128i shuffle1 = _mm_shuffle_epi32(min1, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i min2 = _mm_min_epi32(min1, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(min2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128i res = _mm_min_epi32(min2, shuffle2);
        return _mm_cvtsi128_si32(res);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        __m128i low1 = _mm256_castsi256_si128(v);
        __m128i high1 = _mm256_extracti128_si256(v, 0b1);
        __m128i min1 = _mm_min_epu32(low1, high1);
        __m128i shuffle1 = _mm_shuffle_epi32(min1, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i min2 = _mm_min_epu32(min1, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(min2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128i res = _mm_min_epu32(min2, shuffle2);
        return std::bit_cast<uint32_t>(_mm_cvtsi128_si32(res));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        int64_t sres = std::bit_cast<int64_t>(lanes_v[0]);
        int64_t s1 = std::bit_cast<int64_t>(lanes_v[1]);
        int64_t s2 = std::bit_cast<int64_t>(lanes_v[2]);
        int64_t s3 = std::bit_cast<int64_t>(lanes_v[3]);
        sres = s1 < sres ? s1 : sres;
        sres = s2 < sres ? s2 : sres;
        sres = s3 < sres ? s3 : sres;
        return sres;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        uint64_t res = lanes_v[0];
        res = lanes_v[1] < res ? lanes_v[1] : res;
        res = lanes_v[2] < res ? lanes_v[2] : res;
        res = lanes_v[3] < res ? lanes_v[3] : res;
        return res;
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128d low = _mm256_castpd256_pd128(v);
        __m128d high = _mm256_extractf128_pd(v, 0b1);
        __m128d max1 = _mm_max_pd(low, high);
        __m128d shuffle = _mm_shuffle_pd(max1, max1, 0b01);
        __m128d res = _mm_max_sd(max1, shuffle);

        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            __m256d nan_check = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
            int32_t has_nan = _mm256_movemask_pd(nan_check);
            return has_nan ? QNaN<tag_scalar_t<Tag>> : _mm_cvtsd_f64(res);
        }
        return _mm_cvtsd_f64(res);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i max_vec = _mm_max_epi8(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        max_vec = _mm_max_epi8(max_vec, _mm_shuffle_epi32(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epi8(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epi8(max_vec, _mm_srli_epi32(max_vec, 16));
        max_vec = _mm_max_epi8(max_vec, _mm_srli_epi16(max_vec, 8));
        return static_cast<int8_t>(_mm_extract_epi8(max_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i max_vec = _mm_max_epu8(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        max_vec = _mm_max_epu8(max_vec, _mm_shuffle_epi32(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epu8(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epu8(max_vec, _mm_srli_epi32(max_vec, 16));
        max_vec = _mm_max_epu8(max_vec, _mm_srli_epi16(max_vec, 8));
        return static_cast<uint8_t>(_mm_extract_epi8(max_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i max_vec = _mm_max_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        max_vec = _mm_max_epi16(max_vec, _mm_shuffle_epi32(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epi16(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epi16(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 1, 1, 1)));
        return static_cast<uint16_t>(_mm_extract_epi16(max_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i max_vec = _mm_max_epu16(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        max_vec = _mm_max_epu16(max_vec, _mm_shuffle_epi32(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epu16(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 0, 3, 2)));
        max_vec = _mm_max_epu16(max_vec, _mm_shufflelo_epi16(max_vec, _MM_SHUFFLE(1, 1, 1, 1)));
        return static_cast<uint16_t>(_mm_extract_epi16(max_vec, 0));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i low1 = _mm256_castsi256_si128(v);
        __m128i high1 = _mm256_extracti128_si256(v, 0b1);
        __m128i max1 = _mm_max_epi32(low1, high1);
        __m128i shuffle1 = _mm_shuffle_epi32(max1, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i max2 = _mm_max_epi32(max1, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128i res = _mm_max_epi32(max2, shuffle2);
        return _mm_cvtsi128_si32(res);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        __m128i low1 = _mm256_castsi256_si128(v);
        __m128i high1 = _mm256_extracti128_si256(v, 0b1);
        __m128i max1 = _mm_max_epu32(low1, high1);
        __m128i shuffle1 = _mm_shuffle_epi32(max1, _MM_SHUFFLE(3, 2, 3, 2));
        __m128i max2 = _mm_max_epu32(max1, shuffle1);
        __m128i shuffle2 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(1, 1, 1, 1));
        __m128i res = _mm_max_epu32(max2, shuffle2);
        return std::bit_cast<uint32_t>(_mm_cvtsi128_si32(res));
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        int64_t sres = std::bit_cast<int64_t>(lanes_v[0]);
        int64_t s1 = std::bit_cast<int64_t>(lanes_v[1]);
        int64_t s2 = std::bit_cast<int64_t>(lanes_v[2]);
        int64_t s3 = std::bit_cast<int64_t>(lanes_v[3]);
        sres = s1 > sres ? s1 : sres;
        sres = s2 > sres ? s2 : sres;
        sres = s3 > sres ? s3 : sres;
        return sres;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_256<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        alignas(32) uint64_t lanes_v[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes_v), v);
        uint64_t res = lanes_v[0];
        res = lanes_v[1] > res ? lanes_v[1] : res;
        res = lanes_v[2] > res ? lanes_v[2] : res;
        res = lanes_v[3] > res ? lanes_v[3] : res;
        return res;
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_and_pd(v, _mm256_set1_pd(SignBitClearMask<tag_scalar_t<Tag>>));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_abs_epi8(v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_abs_epi16(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_abs_epi32(v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
        return _mm256_sub_epi64(_mm256_xor_si256(v, sign), sign);
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
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        __m256d mask = _mm256_set1_pd(SignBitMask<tag_scalar_t<Tag>>);
        return _mm256_xor_pd(v, mask);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sub_epi8(_mm256_setzero_si256(), v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sub_epi16(_mm256_setzero_si256(), v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sub_epi32(_mm256_setzero_si256(), v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sub_epi64(_mm256_setzero_si256(), v);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_div_pd(lhs, rhs);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return _mm256_sqrt_pd(v);
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

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return _mm256_round_pd(v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return _mm256_round_pd(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return _mm256_round_pd(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        }
        else
        {
            __m256d sign_bit = _mm256_set1_pd(SignBitMask<tag_scalar_t<Tag>>);
            __m256d half = _mm256_set1_pd(0.5);
            __m256d half_with_sign_bit = _mm256_or_pd(half, _mm256_and_pd(v, sign_bit));
            return _mm256_round_pd(_mm256_add_pd(v, half_with_sign_bit), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NGT_UQ);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NGE_UQ);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NLT_UQ);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_NLE_UQ);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_UNORD_Q);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_and_pd(_mm256_cmp_pd(lhs, lhs, _CMP_UNORD_Q), _mm256_cmp_pd(rhs, rhs, _CMP_UNORD_Q));
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return _mm256_cmp_pd(lhs, rhs, _CMP_ORD_Q);
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256d inf_v = _mm256_set1_pd(Inf<tag_scalar_t<Tag>>);
        return _mm256_or_pd(_mm256_cmp_pd(_mm256_and_pd(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                            _mm256_cmp_pd(_mm256_and_pd(rhs, abs_mask), inf_v, _CMP_LT_OQ));
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

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_256<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        __m256d abs_mask = _mm256_set1_pd(SignBitClearMask<tag_scalar_t<Tag>>);
        __m256d inf_v = _mm256_set1_pd(Inf<tag_scalar_t<Tag>>);
        return _mm256_and_pd(_mm256_cmp_pd(_mm256_and_pd(lhs, abs_mask), inf_v, _CMP_LT_OQ),
                             _mm256_cmp_pd(_mm256_and_pd(rhs, abs_mask), inf_v, _CMP_LT_OQ));
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

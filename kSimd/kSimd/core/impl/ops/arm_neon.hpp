// do not use include guard

#include <arm_neon.h>

#include <cstring>

#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_NEON_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_NEON_END
    #include "shared.hpp"
#endif

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

#define KSIMD_LEVEL_FULL_FP16 \
    ((KSIMD_DYN_DISPATCH_LEVEL == KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16) || (KSIMD_DYN_DISPATCH_LEVEL == KSIMD_DYN_DISPATCH_LEVEL_SVE_FULL_FP16))

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_128 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32 x 4)
        #if !KSIMD_LEVEL_FULL_FP16
        if constexpr (is_tag_float_16bits<Tag>)
        {
            return vec_size::Vec128 / sizeof(float);
        }
        #endif
        return vec_size::Vec128 / sizeof(tag_scalar_t<Tag>);
    }
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        // batch
        template<typename Tag, typename Enable>
        struct batch_type;

        // fake fp16
        #if !KSIMD_LEVEL_FULL_FP16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = float32x4_t;
        };
        // full fp16
        #else
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = float16x8_t;
        };
        #endif

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = float32x4_t;
        };

        // int8
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int8<Tag>>>
        {
            using type = int8x16_t;
        };

        // uint8
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint8<Tag>>>
        {
            using type = uint8x16_t;
        };

        // int16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int16<Tag>>>
        {
            using type = int16x8_t;
        };

        // uint16
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint16<Tag>>>
        {
            using type = uint16x8_t;
        };

        // int32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int32<Tag>>>
        {
            using type = int32x4_t;
        };

        // uint32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint32<Tag>>>
        {
            using type = uint32x4_t;
        };

        // int64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int64<Tag>>>
        {
            using type = int64x2_t;
        };

        // uint64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint64<Tag>>>
        {
            using type = uint64x2_t;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = float64x2_t;
        };

        // mask
        template<typename Tag, typename Enable>
        struct mask_type;

        // fake fp16
        #if !KSIMD_LEVEL_FULL_FP16
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = uint32x4_t;
        };
        // full fp16
        #else
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = uint16x8_t;
        };
        #endif

        // f32
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = uint32x4_t;
        };

        // int8
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int8<Tag>>>
        {
            using type = uint8x16_t;
        };

        // uint8
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint8<Tag>>>
        {
            using type = uint8x16_t;
        };

        // int16
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int16<Tag>>>
        {
            using type = uint16x8_t;
        };

        // uint16
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint16<Tag>>>
        {
            using type = uint16x8_t;
        };

        // int32
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int32<Tag>>>
        {
            using type = uint32x4_t;
        };

        // uint32
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint32<Tag>>>
        {
            using type = uint32x4_t;
        };

        // int64
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_int64<Tag>>>
        {
            using type = uint64x2_t;
        };

        // uint64
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_uint64<Tag>>>
        {
            using type = uint64x2_t;
        };

        // f64
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = uint64x2_t;
        };
    } // namespace detail

    // user type
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion

#pragma region--- any types ---
#pragma region--- any types/load ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f64(reinterpret_cast<const double*>(mem));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s8(mem);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u8(mem);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s16(mem);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u16(mem);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s32(mem);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u32(mem);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s64(mem);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u64(mem);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        float16x4_t f16 = vld1_f16(reinterpret_cast<const float16_t*>(mem));
        return vcvt_f32_f16(f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/load

#pragma region--- any types/store ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f64(reinterpret_cast<double*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s8(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u8(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s16(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u16(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s32(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u32(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s64(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u64(mem, v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        float16x4_t f16 = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<float16_t*>(mem), f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/store

#pragma region--- any types/loadu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f64(reinterpret_cast<const double*>(mem));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s8(mem);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u8(mem);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s16(mem);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u16(mem);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s32(mem);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u32(mem);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_s64(mem);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_u64(mem);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        return load(t, mem);
    }
#endif // FP16
#pragma endregion // any types/loadu

#pragma region--- any types/storeu ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f64(reinterpret_cast<double*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s8(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u8(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s16(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u16(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s32(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u32(mem, v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_s64(mem, v);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_u64(mem, v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        store(t, mem, v);
    }
#endif // FP16
#pragma endregion // any types/storeu

#pragma region--- any types/loadu_partial ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        float32x4_t res = vdupq_n_f32(0.0f);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        float64x2_t res = vdupq_n_f64(0.0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
        {
            if constexpr (is_tag_int8<Tag>)
                return vdupq_n_s8(0);
            else
                return vdupq_n_u8(0);
        }
        if constexpr (is_tag_int8<Tag>)
        {
            int8x16_t res = vdupq_n_s8(0);
            std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
            return res;
        }
        else
        {
            uint8x16_t res = vdupq_n_u8(0);
            std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
            return res;
        }
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        int16x8_t res = vdupq_n_s16(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        uint16x8_t res = vdupq_n_u16(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        int32x4_t res = vdupq_n_s32(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        uint32x4_t res = vdupq_n_u32(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        int64x2_t res = vdupq_n_s64(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        uint64x2_t res = vdupq_n_u64(0);
        if (count == 0) [[unlikely]]
            return res;

        std::memcpy(&res, mem, sizeof(tag_scalar_t<Tag>) * count);
        return res;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return vdupq_n_f32(0.f);

        float16x4_t f16 = vdup_n_f16(0);
        std::memcpy(&f16, mem, sizeof(tag_scalar_t<Tag>) * count);

        // promote to f32x4
        return vcvt_f32_f16(f16);
        #endif
    }
#endif // FP16
#pragma endregion // any types/loadu_partial

#pragma region--- any types/storeu_partial ---
    template<typename Tag>
        requires(!is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        float16x4_t f16 = vcvt_f16_f32(v);
        std::memcpy(mem, &f16, sizeof(tag_scalar_t<Tag>) * count);
        #endif
    }
#endif // FP16
#pragma endregion // any types/storeu_partial

#pragma region--- any types/undefined ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return vdupq_n_f64(0.0);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_s8(0); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_u8(0); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_s16(0); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_u16(0); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_s32(0); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_u32(0); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_s64(0); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return vdupq_n_u64(0); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return undefined(detail::Tag128<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/undefined

#pragma region--- any types/zero ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return vdupq_n_f64(0.0);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_s8(0); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_u8(0); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_s16(0); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_u16(0); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_s32(0); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_u32(0); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_s64(0); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return vdupq_n_u64(0); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return zero(detail::Tag128<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/zero

#pragma region--- any types/set ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return vdupq_n_f32(static_cast<float>(x));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return vdupq_n_f64(static_cast<double>(x));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_s8(x); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_u8(x); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_s16(x); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_u16(x); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_s32(x); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_u32(x); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_s64(x); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return vdupq_n_u64(x); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return set(detail::Tag128<float>{}, static_cast<float>(x));
        #endif
    }
#endif // FP16
#pragma endregion // any types/set

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const float data[4] = { 0.f, 1.f, 2.f, 3.f };
        return vld1q_f32(data);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const double data[2] = { 0.0, 1.0 };
        return vld1q_f64(data);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const int8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vld1q_s8(v);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const uint8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vld1q_u8(v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const int16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vld1q_s16(v);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const uint16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vld1q_u16(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const int32_t v[4] = {0, 1, 2, 3};
        return vld1q_s32(v);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const uint32_t v[4] = {0, 1, 2, 3};
        return vld1q_u32(v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const int64_t v[2] = {0, 1};
        return vld1q_s64(v);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const uint64_t v[2] = {0, 1};
        return vld1q_u64(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{});
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vaddq_f32(sequence(t), base_v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        float64x2_t base_v = vdupq_n_f64(static_cast<double>(base));
        return vaddq_f64(sequence(t), base_v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const int8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vaddq_s8(vdupq_n_s8(base), vld1q_s8(v));
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const uint8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vaddq_u8(vdupq_n_u8(base), vld1q_u8(v));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const int16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vaddq_s16(vdupq_n_s16(base), vld1q_s16(v));
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const uint16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vaddq_u16(vdupq_n_u16(base), vld1q_u16(v));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const int32_t v[4] = {0, 1, 2, 3};
        return vaddq_s32(vdupq_n_s32(base), vld1q_s32(v));
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const uint32_t v[4] = {0, 1, 2, 3};
        return vaddq_u32(vdupq_n_u32(base), vld1q_u32(v));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const int64_t v[2] = {0, 1};
        return vaddq_s64(vdupq_n_s64(base), vld1q_s64(v));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        static const uint64_t v[2] = {0, 1};
        return vaddq_u64(vdupq_n_u64(base), vld1q_u64(v));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{}, static_cast<float>(base));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/sequence ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        float32x4_t stride_v = vdupq_n_f32(static_cast<float>(stride));
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vfmaq_f32(base_v, stride_v, sequence(t));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        float64x2_t stride_v = vdupq_n_f64(static_cast<double>(stride));
        float64x2_t base_v = vdupq_n_f64(static_cast<double>(base));
        return vfmaq_f64(base_v, stride_v, sequence(t));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const int8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vmlaq_s8(vdupq_n_s8(base), vld1q_s8(v), vdupq_n_s8(stride));
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const uint8_t v[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return vmlaq_u8(vdupq_n_u8(base), vld1q_u8(v), vdupq_n_u8(stride));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const int16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vmlaq_s16(vdupq_n_s16(base), vld1q_s16(v), vdupq_n_s16(stride));
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const uint16_t v[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vmlaq_u16(vdupq_n_u16(base), vld1q_u16(v), vdupq_n_u16(stride));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const int32_t v[4] = {0, 1, 2, 3};
        return vmlaq_s32(vdupq_n_s32(base), vld1q_s32(v), vdupq_n_s32(stride));
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        static const uint32_t v[4] = {0, 1, 2, 3};
        return vmlaq_u32(vdupq_n_u32(base), vld1q_u32(v), vdupq_n_u32(stride));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        int64x2_t out = vdupq_n_s64(base);
        return vsetq_lane_s64(base + stride, out, 1);
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        uint64x2_t out = vdupq_n_u64(base);
        return vsetq_lane_u64(base + stride, out, 1);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{}, static_cast<float>(base), static_cast<float>(stride));
        #endif
    }
#endif // FP16
#pragma endregion // any types/sequence

#pragma region--- any types/add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vaddq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vaddq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vaddq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return add(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/add

#pragma region--- any types/sub ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vsubq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vsubq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vsubq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return sub(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/sub

#pragma region--- any types/mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmulq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmulq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vmulq_u32(lhs, rhs); }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        alignas(16) tag_scalar_t<Tag> a[2];
        alignas(16) tag_scalar_t<Tag> b[2];
        alignas(16) tag_scalar_t<Tag> out[2];
        storeu(t, a, lhs);
        storeu(t, b, rhs);
        out[0] = a[0] * b[0];
        out[1] = a[1] * b[1];
        return loadu(t, out);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mul(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul

#pragma region--- any types/mul_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // NEON: v = c + (a * b)
        return vfmaq_f32(c, a, b);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return vfmaq_f64(c, a, b);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept { return add(t, mul(t, a, b), c); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mul_add(detail::Tag128<float>{}, a, b, c);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mul_add

#pragma region--- any types/min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_mask = vornq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
            float32x4_t min_v = vminq_f32(lhs, rhs);
            float32x4_t nan_v = vdupq_n_f32(QNaN<tag_scalar_t<Tag>>);
            return vbslq_f32(nan_mask, nan_v, min_v);
        }
        else
        {
            return vminq_f32(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint64x2_t nan_mask = vornq_u64(vceqq_f64(lhs, lhs), vceqq_f64(rhs, rhs));
            float64x2_t min_v = vminq_f64(lhs, rhs);
            float64x2_t nan_v = vdupq_n_f64(QNaN<tag_scalar_t<Tag>>);
            return vbslq_f64(nan_mask, nan_v, min_v);
        }
        else
        {
            return vminq_f64(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_s8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_u8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_s16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_u16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_s32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vminq_u32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t mask = vcltq_s64(lhs, rhs);
        return vbslq_s64(mask, lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t mask = vcltq_u64(lhs, rhs);
        return vbslq_u64(mask, lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return min<option>(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/min

#pragma region--- any types/max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_mask = vornq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
            float32x4_t max_v = vmaxq_f32(lhs, rhs);
            float32x4_t nan_v = vdupq_n_f32(QNaN<tag_scalar_t<Tag>>);
            return vbslq_f32(nan_mask, nan_v, max_v);
        }
        else
        {
            return vmaxq_f32(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint64x2_t nan_mask = vornq_u64(vceqq_f64(lhs, lhs), vceqq_f64(rhs, rhs));
            float64x2_t max_v = vmaxq_f64(lhs, rhs);
            float64x2_t nan_v = vdupq_n_f64(QNaN<tag_scalar_t<Tag>>);
            return vbslq_f64(nan_mask, nan_v, max_v);
        }
        else
        {
            return vmaxq_f64(lhs, rhs);
        }
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_s8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_u8(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_s16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_u16(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_s32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmaxq_u32(lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t mask = vcgtq_s64(lhs, rhs);
        return vbslq_s64(mask, lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t mask = vcgtq_u64(lhs, rhs);
        return vbslq_u64(mask, lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return max<option>(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/max

#pragma region--- any types/bit_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(v)));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_f64_u32(vmvnq_u32(vreinterpretq_u32_f64(v)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        uint8x16_t u;
        if constexpr (is_tag_int8<Tag>)
            u = vreinterpretq_u8_s8(v);
        else
            u = v;
        uint8x16_t r = vmvnq_u8(u);
        if constexpr (is_tag_int8<Tag>)
            return vreinterpretq_s8_u8(r);
        else
            return r;
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vmvnq_s16(v);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vmvnq_u16(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vmvnq_s32(v);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vmvnq_u32(v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_s64_u32(vmvnq_u32(vreinterpretq_u32_s64(v)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(v)));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_not

#pragma region--- any types/bit_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(lhs), vreinterpretq_u64_f64(rhs)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint8x16_t la, ra;
        if constexpr (is_tag_int8<Tag>)
        {
            la = vreinterpretq_u8_s8(lhs);
            ra = vreinterpretq_u8_s8(rhs);
        }
        else
        {
            la = lhs;
            ra = rhs;
        }
        uint8x16_t r = vandq_u8(la, ra);
        if constexpr (is_tag_int8<Tag>)
            return vreinterpretq_s8_u8(r);
        else
            return r;
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_s16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_s32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_s64_u64(vandq_u64(vreinterpretq_u64_s64(lhs), vreinterpretq_u64_s64(rhs)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and

#pragma region--- any types/bit_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(rhs), vreinterpretq_u32_f32(lhs)));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(rhs), vreinterpretq_u64_f64(lhs)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint8x16_t la, ra;
        if constexpr (is_tag_int8<Tag>)
        {
            la = vreinterpretq_u8_s8(lhs);
            ra = vreinterpretq_u8_s8(rhs);
        }
        else
        {
            la = lhs;
            ra = rhs;
        }
        uint8x16_t r = vbicq_u8(ra, la);
        if constexpr (is_tag_int8<Tag>)
            return vreinterpretq_s8_u8(r);
        else
            return r;
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vbicq_s16(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vbicq_u16(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vbicq_s32(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vbicq_u32(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_s64_u64(vbicq_u64(vreinterpretq_u64_s64(rhs), vreinterpretq_u64_s64(lhs)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vbicq_u64(rhs, lhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_and_not

#pragma region--- any types/bit_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(lhs), vreinterpretq_u64_f64(rhs)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint8x16_t la, ra;
        if constexpr (is_tag_int8<Tag>)
        {
            la = vreinterpretq_u8_s8(lhs);
            ra = vreinterpretq_u8_s8(rhs);
        }
        else
        {
            la = lhs;
            ra = rhs;
        }
        uint8x16_t r = vorrq_u8(la, ra);
        if constexpr (is_tag_int8<Tag>)
            return vreinterpretq_s8_u8(r);
        else
            return r;
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vorrq_s16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vorrq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vorrq_s32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_s64_u64(vorrq_u64(vreinterpretq_u64_s64(lhs), vreinterpretq_u64_s64(rhs)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vorrq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_or

#pragma region--- any types/bit_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(lhs), vreinterpretq_u64_f64(rhs)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint8x16_t la, ra;
        if constexpr (is_tag_int8<Tag>)
        {
            la = vreinterpretq_u8_s8(lhs);
            ra = vreinterpretq_u8_s8(rhs);
        }
        else
        {
            la = lhs;
            ra = rhs;
        }
        uint8x16_t r = veorq_u8(la, ra);
        if constexpr (is_tag_int8<Tag>)
            return vreinterpretq_s8_u8(r);
        else
            return r;
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return veorq_s16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return veorq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return veorq_s32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_s64_u64(veorq_u64(vreinterpretq_u64_s64(lhs), vreinterpretq_u64_s64(rhs)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return veorq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;
#endif // FP16
#pragma endregion // any types/bit_xor

#pragma region--- any types/equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vceqq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vceqq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vceqq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/equal

#pragma region--- any types/not_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vceqq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(lhs, rhs))));
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u8(vceqq_s8(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u8(vceqq_u8(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u16(vceqq_s16(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u16(vceqq_u16(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vceqq_s32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vceqq_u32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_s64(lhs, rhs))));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return bit_not(t, vceqq_u64(lhs, rhs));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/not_equal

#pragma region--- any types/greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgtq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgtq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgtq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return greater(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater

#pragma region--- any types/greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgeq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgeq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcgeq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return greater_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/greater_equal

#pragma region--- any types/less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcltq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcltq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcltq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return less(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less

#pragma region--- any types/less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcleq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcleq_f64(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_s8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_u8(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_s16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_u16(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_s32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_u32(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_s64(lhs, rhs); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return vcleq_u64(lhs, rhs); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return less_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/less_equal

#pragma region--- any types/mask_and ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u64(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_and(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and

#pragma region--- any types/mask_or ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u64(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_or(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_or

#pragma region--- any types/mask_xor ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u64(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u8(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u16(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_xor(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_xor

#pragma region--- any types/mask_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u32(mask);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mask)));
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u8(mask);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u16(mask);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u32(mask);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mask)));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_not(detail::Tag128<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_not

#pragma region--- any types/mask_and_not ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u32(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u64(rhs, lhs);
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u8(rhs, lhs);
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u16(rhs, lhs);
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u32(rhs, lhs);
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u64(rhs, lhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_and_not(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_and_not

#pragma region--- any types/mask_all ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u32(mask) == OneBlock<uint32_t>;
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u32(vreinterpretq_u32_u64(mask)) == OneBlock<uint32_t>;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u8(mask) == OneBlock<uint8_t>;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u16(mask) == OneBlock<uint16_t>;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u32(mask) == OneBlock<uint32_t>;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        return vminvq_u32(vreinterpretq_u32_u64(mask)) == OneBlock<uint32_t>;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_all(detail::Tag128<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_all

#pragma region--- any types/mask_any ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(mask) != 0;
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(vreinterpretq_u32_u64(mask)) != 0;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u8(mask) != 0;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u16(mask) != 0;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(mask) != 0;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(vreinterpretq_u32_u64(mask)) != 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_any(detail::Tag128<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_any

#pragma region--- any types/mask_none ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(mask) == 0;
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(vreinterpretq_u32_u64(mask)) == 0;
    }

    template<typename Tag>
        requires((is_tag_int8<Tag> || is_tag_uint8<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u8(mask) == 0;
    }

    template<typename Tag>
        requires((is_tag_int16<Tag> || is_tag_uint16<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u16(mask) == 0;
    }

    template<typename Tag>
        requires((is_tag_int32<Tag> || is_tag_uint32<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(mask) == 0;
    }

    template<typename Tag>
        requires((is_tag_int64<Tag> || is_tag_uint64<Tag>) && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        return vmaxvq_u32(vreinterpretq_u32_u64(mask)) == 0;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return mask_none(detail::Tag128<float>{}, mask);
        #endif
    }
#endif // FP16
#pragma endregion // any types/mask_none

#pragma region--- any types/if_then_else ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_f32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_f64(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_s8(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_u8(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_s16(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_u16(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_s32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_u32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vreinterpretq_s64_u64(vbslq_u64(_if, vreinterpretq_u64_s64(_then), vreinterpretq_u64_s64(_else)));
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_u64(_if, _then, _else);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return if_then_else(detail::Tag128<float>{}, _if, _then, _else);
        #endif
    }
#endif // FP16
#pragma endregion // any types/if_then_else

#pragma region--- any types/reduce_add ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return vaddvq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return vaddvq_f64(v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        int16x8_t sum16 = vpaddlq_s8(v);
        int32x4_t sum32 = vpaddlq_s16(sum16);
        return vaddvq_s32(sum32);
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        uint16x8_t sum16 = vpaddlq_u8(v);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        return static_cast<int32_t>(vaddvq_u32(sum32));
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // promote to i32
        int32x4_t low  = vmovl_s16(vget_low_s16(v));
        int32x4_t high = vmovl_s16(vget_high_s16(v));
        int32x4_t sum_v = vaddq_s32(low, high);
        return vaddvq_s32(sum_v);
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        uint32x4_t low  = vmovl_u16(vget_low_u16(v));
        uint32x4_t high = vmovl_u16(vget_high_u16(v));
        uint32x4_t sum_v = vaddq_u32(low, high);
        return static_cast<int32_t>(vaddvq_u32(sum_v));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept { return vaddvq_s32(v); }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept { return vaddvq_u32(v); }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept { return vaddvq_s64(v); }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept { return vaddvq_u64(v); }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_add(detail::Tag128<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_add

#pragma region--- any types/reduce_mul ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t res2 = vmul_f32(low, high);
        return vget_lane_f32(res2, 0) * vget_lane_f32(res2, 1);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        float64x1_t low = vget_low_f64(v);
        float64x1_t high = vget_high_f64(v);
        float64x1_t prod = vmul_f64(low, high);
        return vget_lane_f64(prod, 0);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        int16x8_t p1 = vmull_s8(vget_low_s8(v), vget_high_s8(v)); // v[i] * v[i+8]
        int32x4_t p2 = vmull_s16(vget_low_s16(p1), vget_high_s16(p1));
        int32_t a = vgetq_lane_s32(p2, 0);
        int32_t b = vgetq_lane_s32(p2, 1);
        int32_t c = vgetq_lane_s32(p2, 2);
        int32_t d = vgetq_lane_s32(p2, 3);
        return a * b * c * d;
    }

    template<typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        uint16x8_t p16 = vmull_u8(vget_low_u8(v), vget_high_u8(v));
        uint32x4_t p32 = vmull_u16(vget_low_u16(p16), vget_high_u16(p16));
        uint32_t a = vgetq_lane_u32(p32, 0);
        uint32_t b = vgetq_lane_u32(p32, 1);
        uint32_t c = vgetq_lane_u32(p32, 2);
        uint32_t d = vgetq_lane_u32(p32, 3);
        return static_cast<int32_t>(a * b * c * d);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        int32x4_t res32 = vmull_s16(vget_low_s16(v), vget_high_s16(v));
        int32_t a = vgetq_lane_s32(res32, 0);
        int32_t b = vgetq_lane_s32(res32, 1);
        int32_t c = vgetq_lane_s32(res32, 2);
        int32_t d = vgetq_lane_s32(res32, 3);
        return a * b * c * d;
    }

    template<typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(int32_t) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        uint32x4_t res32 = vmull_u16(vget_low_u16(v), vget_high_u16(v));
        uint32_t a = vgetq_lane_u32(res32, 0);
        uint32_t b = vgetq_lane_u32(res32, 1);
        uint32_t c = vgetq_lane_u32(res32, 2);
        uint32_t d = vgetq_lane_u32(res32, 3);
        return static_cast<int32_t>(a * b * c * d);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        int32x2_t low = vget_low_s32(v);
        int32x2_t high = vget_high_s32(v);
        int32x2_t prod = vmul_s32(low, high);
        return vget_lane_s32(prod, 0) * vget_lane_s32(prod, 1);
    }

    template<typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        uint32x2_t low = vget_low_u32(v);
        uint32x2_t high = vget_high_u32(v);
        uint32x2_t prod = vmul_u32(low, high);
        return vget_lane_u32(prod, 0) * vget_lane_u32(prod, 1);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        int64_t lo = vgetq_lane_s64(v, 0);
        int64_t hi = vgetq_lane_s64(v, 1);
        return lo * hi;
    }

    template<typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        uint64_t lo = vgetq_lane_u64(v, 0);
        uint64_t hi = vgetq_lane_u64(v, 1);
        return lo * hi;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_mul(detail::Tag128<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_mul

#pragma region--- any types/reduce_min ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // 检查整个向量是否有 NaN: v != v
            uint32x4_t nan_check = vceqq_f32(v, v);
            // 如果存在 0 (即 NaN)，则返回 NaN
            uint32_t check = vminvq_u32(nan_check);
            if (check == 0)
                return QNaN<tag_scalar_t<Tag>>;
        }
        return vminvq_f32(v);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint64x2_t nan_check = vceqq_f64(v, v);
            if (vminvq_u32(vreinterpretq_u32_u64(nan_check)) == 0)
                return QNaN<tag_scalar_t<Tag>>;
        }
        return vminvq_f64(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_s8(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_u8(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_s16(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_u16(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_s32(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        return vminvq_u32(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        int64_t lo = vgetq_lane_s64(v, 0);
        int64_t hi = vgetq_lane_s64(v, 1);
        return (lo < hi) ? lo : hi;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        uint64_t lo = vgetq_lane_u64(v, 0);
        uint64_t hi = vgetq_lane_u64(v, 1);
        return (lo < hi) ? lo : hi;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_min<option>(detail::Tag128<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_min

#pragma region--- any types/reduce_max ---
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_check = vceqq_f32(v, v);
            if (vminvq_u32(nan_check) == 0)
                return QNaN<tag_scalar_t<Tag>>;
        }
        return vmaxvq_f32(v);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint64x2_t nan_check = vceqq_f64(v, v);
            if (vminvq_u32(vreinterpretq_u32_u64(nan_check)) == 0)
                return QNaN<tag_scalar_t<Tag>>;
        }
        return vmaxvq_f64(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_s8(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint8<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_u8(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_s16(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint16<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_u16(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_s32(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint32<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        return vmaxvq_u32(v);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        int64_t lo = vgetq_lane_s64(v, 0);
        int64_t hi = vgetq_lane_s64(v, 1);
        return (lo > hi) ? lo : hi;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_uint64<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        uint64_t lo = vgetq_lane_u64(v, 0);
        uint64_t hi = vgetq_lane_u64(v, 1);
        return (lo > hi) ? lo : hi;
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_max<option>(detail::Tag128<float>{}, v));
        #endif
    }
#endif // FP16
#pragma endregion // any types/reduce_max
#pragma endregion // any types

#pragma region--- signed ---
#pragma region--- signed/abs ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_f64(v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_s8(v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_s16(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_s32(v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        int64x2_t zero = vdupq_n_s64(0);
        int64x2_t neg_v = vnegq_s64(v);
        uint64x2_t non_negative = vcgeq_s64(v, zero);
        return vreinterpretq_s64_u64(vbslq_u64(non_negative, vreinterpretq_u64_s64(v), vreinterpretq_u64_s64(neg_v)));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return abs(detail::Tag128<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // signed/abs

#pragma region--- signed/neg ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_f64(v);
    }

    template<typename Tag>
        requires(is_tag_int8<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_s8(v);
    }

    template<typename Tag>
        requires(is_tag_int16<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_s16(v);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_s32(v);
    }

    template<typename Tag>
        requires(is_tag_int64<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_s64(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return neg(detail::Tag128<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // signed/neg
#pragma endregion // signed

#pragma region--- floating point ---
#pragma region--- floating point/div ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vdivq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vdivq_f64(lhs, rhs);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return div(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/div

#pragma region--- floating point/sqrt ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return vsqrtq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return vsqrtq_f64(v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return sqrt(detail::Tag128<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/sqrt

#pragma region--- floating point/round ---
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return vrndpq_f32(v); // Positive Infinity
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return vrndmq_f32(v); // Minus Infinity
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return vrndnq_f32(v); // To Nearest (Ties to Even)
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return vrndq_f32(v); // To Zero
        }
        else /* Round (四舍五入) */
        {
            return vrndaq_f32(v);
        }
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)
        {
            return vrndpq_f64(v);
        }
        else if constexpr (mode == RoundingMode::Down)
        {
            return vrndmq_f64(v);
        }
        else if constexpr (mode == RoundingMode::Nearest)
        {
            return vrndnq_f64(v);
        }
        else if constexpr (mode == RoundingMode::ToZero)
        {
            return vrndq_f64(v);
        }
        else
        {
            return vrndaq_f64(v);
        }
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return round<mode>(detail::Tag128<float>{}, v);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/round

#pragma region--- floating point/not_greater ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgtq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vcgtq_f64(lhs, rhs))));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_greater(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater

#pragma region--- floating point/not_greater_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgeq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vcgeq_f64(lhs, rhs))));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_greater_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_greater_equal

#pragma region--- floating point/not_less ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcltq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vcltq_f64(lhs, rhs))));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_less(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less

#pragma region--- floating point/not_less_equal ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcleq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vcleq_f64(lhs, rhs))));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_less_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_less_equal

#pragma region--- floating point/any_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_ok = vceqq_f32(lhs, lhs);
        uint32x4_t r_ok = vceqq_f32(rhs, rhs);
        return vmvnq_u32(vandq_u32(l_ok, r_ok));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t l_ok = vceqq_f64(lhs, lhs);
        uint64x2_t r_ok = vceqq_f64(rhs, rhs);
        return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vandq_u64(l_ok, r_ok))));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return any_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_NaN

#pragma region--- floating point/all_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_nan = vmvnq_u32(vceqq_f32(lhs, lhs));
        uint32x4_t r_nan = vmvnq_u32(vceqq_f32(rhs, rhs));
        return vandq_u32(l_nan, r_nan);
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint64x2_t l_nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(lhs, lhs))));
        uint64x2_t r_nan = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(rhs, rhs))));
        return vandq_u64(l_nan, r_nan);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return all_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_NaN

#pragma region--- floating point/not_NaN ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u64(vceqq_f64(lhs, lhs), vceqq_f64(rhs, rhs));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return not_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/not_NaN

#pragma region--- floating point/any_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<tag_scalar_t<Tag>>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);

        // |x| < Inf 表示有限值
        return vorrq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float64x2_t inf_v = vdupq_n_f64(Inf<tag_scalar_t<Tag>>);
        float64x2_t l_abs = vabsq_f64(lhs);
        float64x2_t r_abs = vabsq_f64(rhs);
        return vorrq_u64(vcltq_f64(l_abs, inf_v), vcltq_f64(r_abs, inf_v));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return any_finite(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/any_finite

#pragma region--- floating point/all_finite ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<tag_scalar_t<Tag>>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);

        return vandq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }

    template<typename Tag>
        requires(is_tag_float_64bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float64x2_t inf_v = vdupq_n_f64(Inf<tag_scalar_t<Tag>>);
        float64x2_t l_abs = vabsq_f64(lhs);
        float64x2_t r_abs = vabsq_f64(rhs);
        return vandq_u64(vcltq_f64(l_abs, inf_v), vcltq_f64(r_abs, inf_v));
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_LEVEL_FULL_FP16

        #else
        return all_finite(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#endif // FP16
#pragma endregion // floating point/all_finite
#pragma endregion // floating point

#pragma region--- float32 only ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return vrecpeq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return vrsqrteq_f32(v);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_LEVEL_FULL_FP16
#undef KSIMD_API

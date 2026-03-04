// do not use include guard

#include <arm_neon.h>

#include <cstring>

#include "kSimd/core/impl/dispatch.hpp"

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_128 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32 x 4)
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16
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
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16
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

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = float32x4_t;
        };

        // mask
        template<typename Tag, typename Enable>
        struct mask_type;

        // fake fp16
        #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16
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

        // f64
        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = uint64x2_t;
        };

        // mask bitset
        template<typename Tag, typename Enable>
        struct mask_bitset_type;

        // 16bits mask
        template<typename Tag>
        struct mask_bitset_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_scalar_16<Tag>>>
        {
            // fake fp16
            #if KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16
            // 其他 16bit 类型使用 u16 ，只有FP16特殊处理
            using type = std::conditional_t<is_tag_float_16bits<Tag>, uint32_t, uint16_t>;
            #else
            using type = uint16_t;
            #endif
        };

        // 32bits mask
        template<typename Tag>
        struct mask_bitset_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_scalar_32<Tag>>>
        {
            using type = uint32_t;
        };

        // 64bits mask
        template<typename Tag>
        struct mask_bitset_type<Tag, std::enable_if_t<is_tag_128<Tag> && is_tag_scalar_64<Tag>>>
        {
            using type = uint64_t;
        };
    } // namespace detail

    // user type
    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;

    template<is_tag Tag>
    using MaskBitset = typename detail::mask_bitset_type<Tag, void>::type;
#pragma endregion

#pragma region--- any types ---

#pragma region--- any types/float32 ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

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
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return vdupq_n_f32(static_cast<float>(x));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const float data[4] = { 0.f, 1.f, 2.f, 3.f };
        return vld1q_f32(data);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vaddq_f32(sequence(t), base_v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        float32x4_t stride_v = vdupq_n_f32(static_cast<float>(stride));
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vfmaq_f32(base_v, stride_v, sequence(t));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vaddq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vsubq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmulq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // NEON: v = c + (a * b)
        return vfmaq_f32(c, a, b);
    }

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

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(v)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(rhs), vreinterpretq_u32_f32(lhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_f32(vreinterpretq_u32_f32(_if), _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vceqq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vceqq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgtq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgeq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcltq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcleq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u32(mask);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vbicq_u32(rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_f32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // [a, b, c, d] -> [a+c, b+d] -> [a+b+c+d]
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t sum2 = vadd_f32(low, high);
        return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t res2 = vmul_f32(low, high);
        return vget_lane_f32(res2, 0) * vget_lane_f32(res2, 1);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag t, Batch<Tag> v) noexcept
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
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag t, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_check = vceqq_f32(v, v);
            if (vminvq_u32(nan_check) == 0)
                return QNaN<tag_scalar_t<Tag>>;
        }
        return vmaxvq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(MaskBitset<Tag>) reduce_mask(Tag, Mask<Tag> mask) noexcept
    {
        // [ 11111, 00000, 11111, 11111 ] // values
        // [ 00001, 00010, 00100, 01000 ] // weights
        // and op
        // [ 00001, 00000, 00100, 01000 ] // 执行 and 操作，原本是0的，还是0，原本不为0的，变成了与weight相同的数
        // reduce add
        // [ 01101 ] // 每一个位执行0 + 1，不会影响其他位，相当于执行 bit or

        uint32x4_t v_weights = {0b1, 0b10, 0b100, 0b1000};
        uint32x4_t masked_weights = vandq_u32(mask, v_weights);
        return vaddvq_u32(masked_weights);
    }
#pragma endregion // any types/float32

#if KSIMD_SUPPORT_NATIVE_FP16
#pragma region--- any types/float16 ---
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        float16x4_t f16 = vld1_f16(reinterpret_cast<const float16_t*>(mem));
        return vcvt_f32_f16(f16);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        float16x4_t f16 = vcvt_f16_f32(v);
        vst1_f16(reinterpret_cast<float16_t*>(mem), f16);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        return load(t, mem);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        store(t, mem, v);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

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

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        float16x4_t f16 = vcvt_f16_f32(v);
        std::memcpy(mem, &f16, sizeof(tag_scalar_t<Tag>) * count);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return undefined(detail::Tag128<float>{});
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return zero(detail::Tag128<float>{});
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return set(detail::Tag128<float>{}, static_cast<float>(x));
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{});
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{}, static_cast<float>(base));
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return sequence(detail::Tag128<float>{}, static_cast<float>(base), static_cast<float>(stride));
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return add(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return sub(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mul(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mul_add(detail::Tag128<float>{}, a, b, c);
        #endif
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return min<option>(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return max<option>(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return greater(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return greater_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return less(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return less_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mask_and(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mask_or(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mask_xor(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mask_not(detail::Tag128<float>{}, mask);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return mask_and_not(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return if_then_else(detail::Tag128<float>{}, _if, _then, _else);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_add(detail::Tag128<float>{}, v));
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_mul(detail::Tag128<float>{}, v));
        #endif
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_min<option>(detail::Tag128<float>{}, v));
        #endif
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return static_cast<tag_scalar_t<Tag>>(reduce_max<option>(detail::Tag128<float>{}, v));
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(MaskBitset<Tag>) reduce_mask(Tag, Mask<Tag> mask) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return static_cast<MaskBitset<Tag>>(reduce_mask(detail::Tag128<float>{}, mask));
        #endif
    }
#pragma endregion // any types/float16
#endif // FP16

#pragma endregion // any types

#pragma region--- signed ---

#pragma region--- signed/float32 ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_f32(v);
    }
#pragma endregion // signed/float32

#if KSIMD_SUPPORT_NATIVE_FP16
#pragma region--- signed/float16 ---
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return abs(detail::Tag128<float>{}, v);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return neg(detail::Tag128<float>{}, v);
        #endif
    }
#pragma endregion // signed/float16
#endif // FP16

#pragma endregion // signed

#pragma region--- floating point ---

#pragma region--- floating point/float32 ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vdivq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return vsqrtq_f32(v);
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag t, Batch<Tag> v) noexcept
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

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgtq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgeq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcltq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcleq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_ok = vceqq_f32(lhs, lhs);
        uint32x4_t r_ok = vceqq_f32(rhs, rhs);
        return vmvnq_u32(vandq_u32(l_ok, r_ok));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_nan = vmvnq_u32(vceqq_f32(lhs, lhs));
        uint32x4_t r_nan = vmvnq_u32(vceqq_f32(rhs, rhs));
        return vandq_u32(l_nan, r_nan);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
    }

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
        requires(is_tag_float_32bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<tag_scalar_t<Tag>>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);

        return vandq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }
#pragma endregion // floating point/float32

#if KSIMD_SUPPORT_NATIVE_FP16
#pragma region--- floating point/float16 ---
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return div(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return sqrt(detail::Tag128<float>{}, v);
        #endif
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return round<mode>(detail::Tag128<float>{}, v);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_greater(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_greater_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_less(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_less_equal(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return any_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return all_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return not_NaN(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return any_finite(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        // full fp16
        #if KSIMD_DYN_DISPATCH_LEVEL >= KSIMD_DYN_DISPATCH_LEVEL_NEON_FULL_FP16

        #else
        return all_finite(detail::Tag128<float>{}, lhs, rhs);
        #endif
    }
#pragma endregion // floating point/float16
#endif // FP16

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

#undef KSIMD_API

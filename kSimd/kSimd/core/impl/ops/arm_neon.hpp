// do not use include guard

#include <arm_neon.h>

#include <cstring>

#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_full_and_fixed128 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        return vec_size::Vec128 / sizeof(tag_scalar_t<Tag>);
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Vec128;
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<is_scalar_type>
        struct batch_type;

        template<>
        struct batch_type<float>
        {
            using type = float32x4_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT32
        template<>
        struct batch_type<std::float32_t>
        {
            using type = float32x4_t;
        };
#endif

        template<>
        struct batch_type<double>
        {
            using type = float64x2_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT64
        template<>
        struct batch_type<std::float64_t>
        {
            using type = float64x2_t;
        };
#endif
    } // namespace detail

    template<typename Tag>
        requires(is_tag_full_and_fixed128<Tag>)
    using Batch = typename detail::batch_type<tag_scalar_t<Tag>>::type;


    namespace detail
    {
        template<is_scalar_type>
        struct mask_type;

        template<>
        struct mask_type<float>
        {
            using type = uint32x4_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT32
        template<>
        struct mask_type<std::float32_t>
        {
            using type = uint32x4_t;
        };
#endif

        template<>
        struct mask_type<double>
        {
            using type = uint64x2_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT64
        template<>
        struct mask_type<std::float64_t>
        {
            using type = uint64x2_t;
        };
#endif
    } // namespace detail

    template<typename Tag>
        requires(is_tag_full_and_fixed128<Tag>)
    using Mask = typename detail::mask_type<tag_scalar_t<Tag>>::type;
#pragma endregion

#pragma region--- any types ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
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
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]]
            return;

        std::memcpy(mem, &v, sizeof(tag_scalar_t<Tag>) * count);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return vdupq_n_f32(static_cast<float>(x));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        static const float data[4] = { 0.f, 1.f, 2.f, 3.f };
        return vld1q_f32(data);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vaddq_f32(sequence(t), base_v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        float32x4_t stride_v = vdupq_n_f32(static_cast<float>(stride));
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        // NEON FMA: res = base + (stride * iota)
        return vfmaq_f32(base_v, stride_v, sequence(t));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vaddq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vsubq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmulq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        // NEON: v = c + (a * b)
        return vfmaq_f32(c, a, b);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
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
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(v)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(rhs), vreinterpretq_u32_f32(lhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        // NEON vbsl: (if) ? then : else
        return vbslq_f32(vreinterpretq_u32_f32(_if), _then, _else);
    }

#if defined(KSIMD_IS_TESTING)
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) test_store_mask(Tag, tag_scalar_t<Tag>* mem, Mask<Tag> mask) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), vreinterpretq_f32_u32(mask));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) test_load_mask(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return vreinterpretq_u32_f32(vld1q_f32(reinterpret_cast<const float*>(mem)));
    }
#endif

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vceqq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vceqq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgtq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcgeq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcltq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vcleq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return vmvnq_u32(mask);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return vbslq_f32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        // [a, b, c, d] -> [a+c, b+d] -> [a+b+c+d]
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t sum2 = vadd_f32(low, high);
        return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t res2 = vmul_f32(low, high);
        return vget_lane_f32(res2, 0) * vget_lane_f32(res2, 1);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
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

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag_float_32bits Tag>
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
#pragma endregion

#pragma region--- signed ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return vabsq_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return vnegq_f32(v);
    }
#pragma endregion

#pragma region--- floating point ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vdivq_f32(lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return vsqrtq_f32(v);
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
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
        else /* Round (Away from zero) */
        {
            return vrndaq_f32(v);
        }
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgtq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcgeq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcltq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vmvnq_u32(vcleq_f32(lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_ok = vceqq_f32(lhs, lhs);
        uint32x4_t r_ok = vceqq_f32(rhs, rhs);
        return vmvnq_u32(vandq_u32(l_ok, r_ok));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        uint32x4_t l_nan = vmvnq_u32(vceqq_f32(lhs, lhs));
        uint32x4_t r_nan = vmvnq_u32(vceqq_f32(rhs, rhs));
        return vandq_u32(l_nan, r_nan);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return vandq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<tag_scalar_t<Tag>>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);

        // |x| < Inf 表示有限值
        return vorrq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<tag_scalar_t<Tag>>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);

        return vandq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }
#pragma endregion

#pragma region--- float32 only ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        // 获取初始近似值 (Estimate)
        float32x4_t estimate = vrecpeq_f32(v);

        // 牛顿迭代
        // 迭代公式: x_{n+1} = x_n * (2 - v * x_n)
        // vrecpsq_f32 执行 (2 - v * x_n)
        float32x4_t iter = vrecpsq_f32(v, estimate);
        return vmulq_f32(estimate, iter);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        // 获取初始近似值 (Estimate)
        float32x4_t estimate = vrsqrteq_f32(v);

        // 牛顿迭代
        // 迭代公式: x_{n+1} = x_n * (3 - v * x_n^2) / 2
        // vrsqrtsq_f32 执行 (3 - v * x_n^2) / 2
        float32x4_t iter = vrsqrtsq_f32(vmulq_f32(estimate, estimate), v);
        return vmulq_f32(estimate, iter);
    }
#pragma endregion
} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

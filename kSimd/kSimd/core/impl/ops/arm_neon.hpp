// do not use include guard

#include <arm_neon.h>

#include <cstring>

#include "op.hpp"
#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- traits ---
    template<is_scalar_type S>
    struct Traits
    {
        using _scalar_type = S;
        static constexpr size_t _lanes = vec_size::Vec128 / sizeof(S);
    };

    template<is_scalar_type S>
    constexpr size_t lanes(Traits<S>) noexcept
    {
        return Traits<S>::_lanes;
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

    template<is_scalar_type S>
    using Batch = typename detail::batch_type<S>::type;


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

    template<is_scalar_type S>
    using Mask = typename detail::mask_type<S>::type;
#pragma endregion

#pragma region--- any types ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) load(Traits<S>, const S* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) store(Traits<S>, S* mem, Batch<S> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu(Traits<S>, const S* mem) noexcept
    {
        return vld1q_f32(reinterpret_cast<const float*>(mem));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu(Traits<S>, S* mem, Batch<S> v) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) loadu_partial(Traits<S> t, const S* mem, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        float32x4_t res = vdupq_n_f32(0.0f);
        if (count == 0) [[unlikely]] return res;
        
        std::memcpy(&res, mem, sizeof(S) * count);
        return res;
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) storeu_partial(Traits<S> t, S* mem, Batch<S> v, size_t count) noexcept
    {
        constexpr size_t L = lanes(t);
        count = count > L ? L : count;
        if (count == 0) [[unlikely]] return;

        std::memcpy(mem, &v, sizeof(S) * count);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) undefined(Traits<S>) noexcept
    {
        return vdupq_n_f32(0.0f); 
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) zero(Traits<S>) noexcept
    {
        return vdupq_n_f32(0.0f);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) set(Traits<S>, S x) noexcept
    {
        return vdupq_n_f32(static_cast<float>(x));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S>) noexcept
    {
        static const float data[4] = { 0.f, 1.f, 2.f, 3.f };
        return vld1q_f32(data);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S> t, S base) noexcept
    {
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        return vaddq_f32(sequence(t), base_v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sequence(Traits<S> t, S base, S stride) noexcept
    {
        float32x4_t stride_v = vdupq_n_f32(static_cast<float>(stride));
        float32x4_t base_v = vdupq_n_f32(static_cast<float>(base));
        // NEON FMA: res = base + (stride * iota)
        return vfmaq_f32(base_v, stride_v, sequence(t));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) add(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vaddq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sub(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vsubq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmulq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) mul_add(Traits<S>, Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        // NEON: v = c + (a * b)
        return vfmaq_f32(c, a, b);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) min(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_mask = vornq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
            float32x4_t min_v = vminq_f32(lhs, rhs);
            float32x4_t nan_v = vdupq_n_f32(QNaN<S>);
            return vbslq_f32(nan_mask, nan_v, min_v);
        }
        else
        {
            return vminq_f32(lhs, rhs);
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) max(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_mask = vornq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
            float32x4_t max_v = vmaxq_f32(lhs, rhs);
            float32x4_t nan_v = vdupq_n_f32(QNaN<S>);
            return vbslq_f32(nan_mask, nan_v, max_v);
        }
        else
        {
            return vmaxq_f32(lhs, rhs);
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_not(Traits<S>, Batch<S> v) noexcept
    {
        return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(v)));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_and_not(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(rhs), vreinterpretq_u32_f32(lhs)));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_or(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_xor(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(lhs), vreinterpretq_u32_f32(rhs)));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) bit_if_then_else(Traits<S>, Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        // NEON vbsl: (if) ? then : else
        return vbslq_f32(vreinterpretq_u32_f32(_if), _then, _else);
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type_float_32bits S>
    KSIMD_API(void) test_store_mask(Traits<S>, S* mem, Mask<S> mask) noexcept
    {
        vst1q_f32(reinterpret_cast<float*>(mem), vreinterpretq_f32_u32(mask));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) test_load_mask(Traits<S>, const S* mem) noexcept
    {
        return vreinterpretq_u32_f32(vld1q_f32(reinterpret_cast<const float*>(mem)));
    }
#endif

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vceqq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmvnq_u32(vceqq_f32(lhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vcgtq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) greater_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vcgeq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vcltq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) less_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vcleq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_and(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return vandq_u32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_or(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return vorrq_u32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_xor(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return veorq_u32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) mask_not(Traits<S>, Mask<S> mask) noexcept
    {
        return vmvnq_u32(mask);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) if_then_else(Traits<S>, Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return vbslq_f32(_if, _then, _else);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_add(Traits<S>, Batch<S> v) noexcept
    {
        // [a, b, c, d] -> [a+c, b+d] -> [a+b+c+d]
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t sum2 = vadd_f32(low, high);
        return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_mul(Traits<S>, Batch<S> v) noexcept
    {
        float32x2_t low = vget_low_f32(v);
        float32x2_t high = vget_high_f32(v);
        float32x2_t res2 = vmul_f32(low, high);
        return vget_lane_f32(res2, 0) * vget_lane_f32(res2, 1);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_min(Traits<S> t, Batch<S> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            // 检查整个向量是否有 NaN: v != v
            uint32x4_t nan_check = vceqq_f32(v, v); 
            // 如果存在 0 (即 NaN)，则返回 NaN
            uint32_t check = vminvq_u32(nan_check);
            if (check == 0) return QNaN<S>;
        }
        return vminvq_f32(v);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type_float_32bits S>
    KSIMD_API(S) reduce_max(Traits<S> t, Batch<S> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            uint32x4_t nan_check = vceqq_f32(v, v);
            if (vminvq_u32(nan_check) == 0) return QNaN<S>;
        }
        return vmaxvq_f32(v);
    }
#pragma endregion

#pragma region--- signed ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) abs(Traits<S>, Batch<S> v) noexcept
    {
        return vabsq_f32(v);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) neg(Traits<S>, Batch<S> v) noexcept
    {
        return vnegq_f32(v);
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) div(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vdivq_f32(lhs, rhs);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) sqrt(Traits<S>, Batch<S> v) noexcept
    {
        return vsqrtq_f32(v);
    }

    template<RoundingMode mode, is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) round(Traits<S> t, Batch<S> v) noexcept
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
            return vrndq_f32(v);  // To Zero
        }
        else /* Round (Away from zero) */
        {
            return vrndaq_f32(v); 
        }
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmvnq_u32(vcgtq_f32(lhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_greater_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmvnq_u32(vcgeq_f32(lhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmvnq_u32(vcltq_f32(lhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_less_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vmvnq_u32(vcleq_f32(lhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        uint32x4_t l_ok = vceqq_f32(lhs, lhs);
        uint32x4_t r_ok = vceqq_f32(rhs, rhs);
        return vmvnq_u32(vandq_u32(l_ok, r_ok));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        uint32x4_t l_nan = vmvnq_u32(vceqq_f32(lhs, lhs));
        uint32x4_t r_nan = vmvnq_u32(vceqq_f32(rhs, rhs));
        return vandq_u32(l_nan, r_nan);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) not_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return vandq_u32(vceqq_f32(lhs, lhs), vceqq_f32(rhs, rhs));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) any_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<S>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);
        
        // |x| < Inf 表示有限值
        return vorrq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Mask<S>) all_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        float32x4_t inf_v = vdupq_n_f32(Inf<S>);
        float32x4_t l_abs = vabsq_f32(lhs);
        float32x4_t r_abs = vabsq_f32(rhs);
        
        return vandq_u32(vcltq_f32(l_abs, inf_v), vcltq_f32(r_abs, inf_v));
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Traits<S>, Batch<S> v) noexcept
    {
        // 获取初始近似值 (Estimate)
        float32x4_t estimate = vrecpeq_f32(v);

        // 牛顿迭代
        // 迭代公式: x_{n+1} = x_n * (2 - v * x_n)
        // vrecpsq_f32 执行 (2 - v * x_n)
        float32x4_t iter = vrecpsq_f32(v, estimate);
        return vmulq_f32(estimate, iter);
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Traits<S>, Batch<S> v) noexcept
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
}

#undef KSIMD_API

// do not use include guard

#include <arm_sve.h>

#include <array>
#include <cstring>

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

// 复用 NEON 的逻辑，实现 Fixed128Tag
#include "arm_neon.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_scalable_full Tag>
    KSIMD_API(size_t) lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32)
        #if KSIMD_DYN_DISPATCH_LEVEL != KSIMD_DYN_DISPATCH_LEVEL_SVE_FULL_FP16
        if constexpr (is_tag_float_16bits<Tag>)
        {
            return static_cast<size_t>(svcntw());
        }
        #endif

        constexpr size_t len = sizeof(tag_scalar_t<Tag>);

        static_assert(len == 1 || len == 2 || len == 4 || len == 8, "sizeof(scalar type) can only equal to 1, 2, 4, 8");

        if constexpr (len == 1) return static_cast<size_t>(svcntb());
        if constexpr (len == 2) return static_cast<size_t>(svcnth());
        if constexpr (len == 4) return static_cast<size_t>(svcntw());
        return static_cast<size_t>(svcntd()); /* if constexpr (len == 8) */
    }

#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t SVE_MAX_LANES_F32 = 64; // 2048 / 32

        template<typename Tag, typename Enable>
        struct batch_type;

        // fake fp16 => use f32 vector
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_float_16bits<Tag>>>
        {
            using type = svfloat32_t;
        };

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = svfloat32_t;
        };

        // int32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_int32<Tag>>>
        {
            using type = svint32_t;
        };

        template<typename Tag, typename Enable>
        struct mask_type;

        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag>>>
        {
            using type = svbool_t;
        };
    } // namespace detail

    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion

#pragma region--- any types ---
#pragma region--- any types/float32 ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return svld1_f32(svptrue_b32(), reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        svst1_f32(svptrue_b32(), reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return svld1_f32(svptrue_b32(), reinterpret_cast<const float*>(mem));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        svst1_f32(svptrue_b32(), reinterpret_cast<float*>(mem), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        count = count > lanes(t) ? lanes(t) : count;
        svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
        return svld1_f32(pg, mem);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        count = count > lanes(t) ? lanes(t) : count;
        svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
        svst1_f32(pg, mem, v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return svdup_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return svdup_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return svdup_n_f32(static_cast<float>(x));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) tmp[i] = static_cast<float>(i);
        return svld1_f32(svptrue_b32(), tmp.data());
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        return svadd_f32_x(svptrue_b32(), sequence(t), svdup_n_f32(static_cast<float>(base)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        svfloat32_t idx = sequence(t);
        svfloat32_t s = svdup_n_f32(static_cast<float>(stride));
        svfloat32_t b = svdup_n_f32(static_cast<float>(base));
        return svmla_f32_x(svptrue_b32(), b, idx, s);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svadd_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svsub_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svmul_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return svmla_f32_x(svptrue_b32(), c, a, b);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t m = svmin_f32_x(svptrue_b32(), lhs, rhs);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            svbool_t nan_m = any_NaN(t, lhs, rhs);
            svfloat32_t nan_v = svdup_n_f32(QNaN<tag_scalar_t<Tag>>);
            return svsel_f32(nan_m, nan_v, m);
        }
        return m;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t m = svmax_f32_x(svptrue_b32(), lhs, rhs);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            svbool_t nan_m = any_NaN(t, lhs, rhs);
            svfloat32_t nan_v = svdup_n_f32(QNaN<tag_scalar_t<Tag>>);
            return svsel_f32(nan_m, nan_v, m);
        }
        return m;
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        svuint32_t x = svreinterpret_u32_f32(v);
        svuint32_t all1 = svdup_n_u32(UINT32_MAX);
        return svreinterpret_f32_u32(sveor_u32_x(svptrue_b32(), x, all1));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_f32_u32(
            svand_u32_x(svptrue_b32(), svreinterpret_u32_f32(lhs), svreinterpret_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svuint32_t l = svreinterpret_u32_f32(lhs);
        svuint32_t r = svreinterpret_u32_f32(rhs);
        svuint32_t all1 = svdup_n_u32(UINT32_MAX);
        svuint32_t n = sveor_u32_x(svptrue_b32(), l, all1);
        return svreinterpret_f32_u32(svand_u32_x(svptrue_b32(), n, r));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_f32_u32(
            svorr_u32_x(svptrue_b32(), svreinterpret_u32_f32(lhs), svreinterpret_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_f32_u32(
            sveor_u32_x(svptrue_b32(), svreinterpret_u32_f32(lhs), svreinterpret_u32_f32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpeq_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpne_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpgt_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpge_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmplt_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmple_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svorr_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return sveor_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> m) noexcept
    {
        return svnot_z(svptrue_b32(), m);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(), svnot_z(svptrue_b32(), lhs), rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return !svptest_any(pg, svnot_b_z(pg, mask));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        auto pg = svptrue_b32();
        return !svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return svsel_f32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return svaddv_f32(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag t, Batch<Tag> v) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        float r = 1.0f;
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) r *= tmp[i];
        return r;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag t, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
            svst1_f32(svptrue_b32(), tmp.data(), v);
            const size_t l = lanes(t);
            for (size_t i = 0; i < l; ++i)
            {
                if (tmp[i] != tmp[i]) return QNaN<tag_scalar_t<Tag>>;
            }
        }
        return svminv_f32(svptrue_b32(), v);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag t, Batch<Tag> v) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
            svst1_f32(svptrue_b32(), tmp.data(), v);
            const size_t l = lanes(t);
            for (size_t i = 0; i < l; ++i)
            {
                if (tmp[i] != tmp[i]) return QNaN<tag_scalar_t<Tag>>;
            }
        }
        return svmaxv_f32(svptrue_b32(), v);
    }
#pragma endregion // any types/float32

#pragma region--- any types/int32 ---
    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return svld1_s32(svptrue_b32(), mem);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        svst1_s32(svptrue_b32(), mem, v);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        return svld1_s32(svptrue_b32(), mem);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) storeu(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        svst1_s32(svptrue_b32(), mem, v);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
        return svld1_s32(pg, mem);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(0), static_cast<uint64_t>(count));
        svst1_s32(pg, mem, v);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept { return svdup_n_s32(0); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept { return svdup_n_s32(0); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept { return svdup_n_s32(x); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept { return svindex_s32(0, 1); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return svadd_n_s32_x(svptrue_b32(), svindex_s32(0, 1), base);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        return svindex_s32(base, stride);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svadd_s32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svsub_s32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svmul_s32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag t, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return add(t, mul(t, a, b), c);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svmin_s32_x(svptrue_b32(), lhs, rhs);
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svmax_s32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        svuint32_t x = svreinterpret_u32_s32(v);
        svuint32_t all1 = svdup_n_u32(UINT32_MAX);
        return svreinterpret_s32_u32(sveor_u32_x(svptrue_b32(), x, all1));
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_s32_u32(svand_u32_x(svptrue_b32(), svreinterpret_u32_s32(lhs), svreinterpret_u32_s32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svuint32_t l = svreinterpret_u32_s32(lhs);
        svuint32_t r = svreinterpret_u32_s32(rhs);
        svuint32_t all1 = svdup_n_u32(UINT32_MAX);
        svuint32_t n = sveor_u32_x(svptrue_b32(), l, all1);
        return svreinterpret_s32_u32(svand_u32_x(svptrue_b32(), n, r));
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_s32_u32(svorr_u32_x(svptrue_b32(), svreinterpret_u32_s32(lhs), svreinterpret_u32_s32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svreinterpret_s32_u32(sveor_u32_x(svptrue_b32(), svreinterpret_u32_s32(lhs), svreinterpret_u32_s32(rhs)));
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmpeq_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmpne_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmpgt_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmpge_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmplt_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept { return svcmple_s32(svptrue_b32(), lhs, rhs); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svand_b_z(pg, lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svorr_b_z(pg, lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        svbool_t pg = svptrue_b32();
        return sveor_b_z(pg, lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svnot_b_z(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svbic_b_z(svptrue_b32(), rhs, lhs);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return !svptest_any(pg, svnot_b_z(pg, mask));
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_int32<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        auto pg = svptrue_b32();
        return !svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return svsel_s32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept { return svaddv_s32(svptrue_b32(), v); }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag t, Batch<Tag> v) noexcept
    {
        std::array<int32_t, detail::SVE_MAX_LANES_F32> tmp{};
        store(t, tmp.data(), v);
        const size_t l = lanes(t);
        int32_t r = 1;
        for (size_t i = 0; i < l; ++i) r *= tmp[i];
        return r;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept { return svminv_s32(svptrue_b32(), v); }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept { return svmaxv_s32(svptrue_b32(), v); }
#pragma endregion // any types/int32

#if KSIMD_SUPPORT_NATIVE_FP16
#pragma region--- any types/float16 ---
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) tmp[i] = static_cast<float>(mem[i]);
        return svld1_f32(svptrue_b32(), tmp.data());
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) store(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) mem[i] = static_cast<tag_scalar_t<Tag>>(tmp[i]);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        return load(t, mem);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) storeu(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        store(t, mem, v);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag t, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        const size_t l = lanes(t);
        count = count > l ? l : count;
        for (size_t i = 0; i < count; ++i) tmp[i] = static_cast<float>(mem[i]);
        return svld1_f32(svptrue_b32(), tmp.data());
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(void) storeu_partial(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        const size_t l = lanes(t);
        count = count > l ? l : count;
        for (size_t i = 0; i < count; ++i) mem[i] = static_cast<tag_scalar_t<Tag>>(tmp[i]);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return svdup_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        return svdup_n_f32(0.0f);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return svdup_n_f32(static_cast<float>(x));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) tmp[i] = static_cast<float>(i);
        return svld1_f32(svptrue_b32(), tmp.data());
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base) noexcept
    {
        return svadd_f32_x(svptrue_b32(), sequence(t), svdup_n_f32(static_cast<float>(base)));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag t, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        svfloat32_t idx = sequence(t);
        svfloat32_t s = svdup_n_f32(static_cast<float>(stride));
        svfloat32_t b = svdup_n_f32(static_cast<float>(base));
        return svmla_f32_x(svptrue_b32(), b, idx, s);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svadd_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svsub_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svmul_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return svmla_f32_x(svptrue_b32(), c, a, b);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) min(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t m = svmin_f32_x(svptrue_b32(), lhs, rhs);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            svbool_t nan_m = any_NaN(t, lhs, rhs);
            svfloat32_t nan_v = svdup_n_f32(QNaN<float>);
            return svsel_f32(nan_m, nan_v, m);
        }
        return m;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) max(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t m = svmax_f32_x(svptrue_b32(), lhs, rhs);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            svbool_t nan_m = any_NaN(t, lhs, rhs);
            svfloat32_t nan_v = svdup_n_f32(QNaN<float>);
            return svsel_f32(nan_m, nan_v, m);
        }
        return m;
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag>) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag>, Batch<Tag>) noexcept = delete;

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpeq_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpne_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpgt_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmpge_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmplt_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svcmple_f32(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svorr_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return sveor_z(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> m) noexcept
    {
        return svnot_z(svptrue_b32(), m);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) mask_and_not(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(), svnot_z(svptrue_b32(), lhs), rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_all(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return !svptest_any(pg, svnot_b_z(pg, mask));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_any(Tag, Mask<Tag> mask) noexcept
    {
        svbool_t pg = svptrue_b32();
        return svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(bool) mask_none(Tag, Mask<Tag> mask) noexcept
    {
        auto pg = svptrue_b32();
        return !svptest_any(pg, mask);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return svsel_f32(_if, _then, _else);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag t, Batch<Tag> v) noexcept
    {
        float s = svaddv_f32(svptrue_b32(), v);
        return static_cast<tag_scalar_t<Tag>>(s);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag t, Batch<Tag> v) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        float r = 1.0f;
        const size_t l = lanes(t);
        for (size_t i = 0; i < l; ++i) r *= tmp[i];
        return static_cast<tag_scalar_t<Tag>>(r);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag t, Batch<Tag> v) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        const size_t l = lanes(t);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            for (size_t i = 0; i < l; ++i)
            {
                if (tmp[i] != tmp[i]) return static_cast<tag_scalar_t<Tag>>(QNaN<float>);
            }
        }
        float m = tmp[0];
        for (size_t i = 1; i < l; ++i) m = m < tmp[i] ? m : tmp[i];
        return static_cast<tag_scalar_t<Tag>>(m);
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag t, Batch<Tag> v) noexcept
    {
        std::array<float, detail::SVE_MAX_LANES_F32> tmp{};
        svst1_f32(svptrue_b32(), tmp.data(), v);
        const size_t l = lanes(t);
        if constexpr (option == FloatMinMaxOption::CheckNaN)
        {
            for (size_t i = 0; i < l; ++i)
            {
                if (tmp[i] != tmp[i]) return static_cast<tag_scalar_t<Tag>>(QNaN<float>);
            }
        }
        float m = tmp[0];
        for (size_t i = 1; i < l; ++i) m = m > tmp[i] ? m : tmp[i];
        return static_cast<tag_scalar_t<Tag>>(m);
    }
#pragma endregion // any types/float16
#endif

#pragma endregion // any types

#pragma region--- signed ---
    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return svabs_s32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_scalable_full<Tag> && is_tag_int32<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return svneg_s32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return svabs_f32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return svneg_f32_x(svptrue_b32(), v);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return svabs_f32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return svneg_f32_x(svptrue_b32(), v);
    }
#endif
#pragma endregion

#pragma region--- floating point ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svdiv_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return svsqrt_f32_x(svptrue_b32(), v);
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)          return svrintp_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::Down)   return svrintm_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::Nearest)return svrintn_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::ToZero) return svrintz_f32_x(svptrue_b32(), v);
        else                                              return svrinta_f32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmpgt_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmpge_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmplt_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmple_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svbool_t l_ok = svcmpeq_f32(svptrue_b32(), lhs, lhs);
        svbool_t r_ok = svcmpeq_f32(svptrue_b32(), rhs, rhs);
        return svnot_z(svptrue_b32(), svand_z(svptrue_b32(), l_ok, r_ok));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svbool_t l_nan = svcmpne_f32(svptrue_b32(), lhs, lhs);
        svbool_t r_nan = svcmpne_f32(svptrue_b32(), rhs, rhs);
        return svand_z(svptrue_b32(), l_nan, r_nan);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(),
            svcmpeq_f32(svptrue_b32(), lhs, lhs),
            svcmpeq_f32(svptrue_b32(), rhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t inf_v = svdup_n_f32(Inf<tag_scalar_t<Tag>>);
        svbool_t l = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), lhs), inf_v);
        svbool_t r = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), rhs), inf_v);
        return svorr_z(svptrue_b32(), l, r);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t inf_v = svdup_n_f32(Inf<tag_scalar_t<Tag>>);
        svbool_t l = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), lhs), inf_v);
        svbool_t r = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), rhs), inf_v);
        return svand_z(svptrue_b32(), l, r);
    }

#if KSIMD_SUPPORT_NATIVE_FP16
    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svdiv_f32_x(svptrue_b32(), lhs, rhs);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return svsqrt_f32_x(svptrue_b32(), v);
    }

    template<RoundingMode mode, typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        if constexpr (mode == RoundingMode::Up)          return svrintp_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::Down)   return svrintm_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::Nearest)return svrintn_f32_x(svptrue_b32(), v);
        else if constexpr (mode == RoundingMode::ToZero) return svrintz_f32_x(svptrue_b32(), v);
        else                                              return svrinta_f32_x(svptrue_b32(), v);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmpgt_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmpge_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmplt_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svnot_z(svptrue_b32(), svcmple_f32(svptrue_b32(), lhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svbool_t l_ok = svcmpeq_f32(svptrue_b32(), lhs, lhs);
        svbool_t r_ok = svcmpeq_f32(svptrue_b32(), rhs, rhs);
        return svnot_z(svptrue_b32(), svand_z(svptrue_b32(), l_ok, r_ok));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svbool_t l_nan = svcmpne_f32(svptrue_b32(), lhs, lhs);
        svbool_t r_nan = svcmpne_f32(svptrue_b32(), rhs, rhs);
        return svand_z(svptrue_b32(), l_nan, r_nan);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return svand_z(svptrue_b32(),
            svcmpeq_f32(svptrue_b32(), lhs, lhs),
            svcmpeq_f32(svptrue_b32(), rhs, rhs));
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t inf_v = svdup_n_f32(Inf<float>);
        svbool_t l = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), lhs), inf_v);
        svbool_t r = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), rhs), inf_v);
        return svorr_z(svptrue_b32(), l, r);
    }

    template<typename Tag>
        requires(is_tag_float_16bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        svfloat32_t inf_v = svdup_n_f32(Inf<float>);
        svbool_t l = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), lhs), inf_v);
        svbool_t r = svcmplt_f32(svptrue_b32(), svabs_f32_x(svptrue_b32(), rhs), inf_v);
        return svand_z(svptrue_b32(), l, r);
    }
#endif
#pragma endregion

#pragma region--- float32 only ---
    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return svrecpe_f32(v);
    }

    template<typename Tag>
        requires(is_tag_float_32bits<Tag> && is_tag_scalable_full<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return svrsqrte_f32(v);
    }
#pragma endregion
}

#undef KSIMD_API

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "math.cpp"
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        KSIMD_DYN_FUNC_ATTR void kernel(
            const double* KSIMD_RESTRICT src,
                  double* KSIMD_RESTRICT dst,
            const size_t                 size
        ) noexcept
        {
            namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;
            using f64 = KSIMD_DYN_OP(double);

            const f64::batch_t c10 = f64::set(0.123);
            const f64::batch_t c9 = f64::set(-0.456);
            const f64::batch_t c8 = f64::set(0.789);
            const f64::batch_t c7 = f64::set(-0.101);
            const f64::batch_t c6 = f64::set(0.234);
            const f64::batch_t c5 = f64::set(-0.567);
            const f64::batch_t c4 = f64::set(0.890);
            const f64::batch_t c3 = f64::set(-1.234);
            const f64::batch_t c2 = f64::set(0.555);
            const f64::batch_t c1 = f64::set(1.999);
            const f64::batch_t c0 = f64::set(-0.777);

            size_t i = 0;
            for (; i + f64::Lanes <= size; i += f64::Lanes)
            {
                f64::batch_t x = f64::load(src + i);

                f64::batch_t res = ext::math::safe_clamp(c10, x, c9);
                res = ext::math::lerp(res, x, c8);
                res = ext::math::lerp(res, x, c7);
                res = ext::math::lerp(res, x, c6);
                res = ext::math::lerp(res, x, c5);
                res = ext::math::safe_clamp(res, x, c4);
                res = ext::math::safe_clamp(res, x, c3);
                res = ext::math::safe_clamp(res, x, c2);
                res = ext::math::safe_clamp(res, x, c1);
                res = ext::math::safe_clamp(res, x, c0);

                f64::store(dst + i, res);
            }

            // 尾处理
            if (const size_t tail = size - i; tail > 0)
            {
                const f64::mask_t mask = f64::mask_from_lanes(static_cast<unsigned int>(tail));
                f64::batch_t x = f64::mask_load(src + i, mask);

                f64::batch_t res = ext::math::safe_clamp(c10, x, c9);
                res = ext::math::lerp(res, x, c8);
                res = ext::math::lerp(res, x, c7);
                res = ext::math::lerp(res, x, c6);
                res = ext::math::lerp(res, x, c5);
                res = ext::math::lerp(res, x, c4);
                res = ext::math::safe_clamp(res, x, c3);
                res = ext::math::safe_clamp(res, x, c2);
                res = ext::math::safe_clamp(res, x, c1);
                res = ext::math::safe_clamp(res, x, c0);

                f64::mask_store(dst + i, res, mask);
            }
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    KSIMD_DYN_DISPATCH_FUNC(kernel)
    void kernel(
        const double* KSIMD_RESTRICT src,
              double* KSIMD_RESTRICT dst,
        const size_t                 size
    ) noexcept
    {
        KSIMD_DYN_CALL(kernel)(src, dst, size);
    }
} // namespace MyNamespace
#endif

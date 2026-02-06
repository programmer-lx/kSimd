#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "math.cpp"
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/base_op.hpp>
#include <kSimd_extension/vmath.hpp>

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        KSIMD_DYN_FUNC_ATTR void kernel(
            const float* KSIMD_RESTRICT src,
                  float* KSIMD_RESTRICT dst,
            const size_t                 size
        ) noexcept
        {
            namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;
            using f32 = KSIMD_DYN_BASE_OP(float);

            const f32::batch_t c10 = f32::set(0.123);
            const f32::batch_t c9 = f32::set(-0.456);
            const f32::batch_t c8 = f32::set(0.789);
            const f32::batch_t c7 = f32::set(-0.101);
            const f32::batch_t c6 = f32::set(0.234);
            const f32::batch_t c5 = f32::set(-0.567);
            const f32::batch_t c4 = f32::set(0.890);
            const f32::batch_t c3 = f32::set(-1.234);
            const f32::batch_t c2 = f32::set(0.555);
            const f32::batch_t c1 = f32::set(1.999);
            const f32::batch_t c0 = f32::set(-0.777);

            size_t i = 0;
            for (; i + f32::TotalLanes <= size; i += f32::TotalLanes)
            {
                f32::batch_t x = f32::load(src + i);

                f32::batch_t res = ext::vmath::clamp<f32>(c10, x, c9);
                res = ext::vmath::lerp<f32>(res, x, c8);
                res = ext::vmath::lerp<f32>(res, x, c7);
                res = ext::vmath::lerp<f32>(res, x, c6);
                res = ext::vmath::lerp<f32>(res, x, c5);
                res = ext::vmath::clamp<f32>(res, x, c4);
                res = ext::vmath::clamp<f32>(res, x, c3);
                res = ext::vmath::clamp<f32>(res, x, c2);
                res = ext::vmath::clamp<f32>(res, x, c1);
                res = ext::vmath::clamp<f32>(res, x, c0);

                f32::store(dst + i, res);
            }

            // 尾处理
            if (const size_t tail = size - i; tail > 0)
            {
                f32::batch_t x = f32::load_partial(src + i, tail);

                f32::batch_t res = ext::vmath::clamp<f32>(c10, x, c9);
                res = ext::vmath::lerp<f32>(res, x, c8);
                res = ext::vmath::lerp<f32>(res, x, c7);
                res = ext::vmath::lerp<f32>(res, x, c6);
                res = ext::vmath::lerp<f32>(res, x, c5);
                res = ext::vmath::lerp<f32>(res, x, c4);
                res = ext::vmath::clamp<f32>(res, x, c3);
                res = ext::vmath::clamp<f32>(res, x, c2);
                res = ext::vmath::clamp<f32>(res, x, c1);
                res = ext::vmath::clamp<f32>(res, x, c0);

                f32::store_partial(dst + i, res, tail);
            }
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    KSIMD_DYN_DISPATCH_FUNC(kernel)
    void kernel(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                 size
    ) noexcept
    {
        KSIMD_DYN_CALL(kernel)(src, dst, size);
    }
} // namespace MyNamespace
#endif

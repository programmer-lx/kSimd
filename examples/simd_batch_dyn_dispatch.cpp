#include <string>
#include <vector>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_batch_dyn_dispatch.cpp"
#include <kSimd/core/dispatch_this_file.hpp>

#include <kSimd/core/aligned_allocate.hpp>
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/dispatch_vmath.hpp>

#include "utils.hpp"

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        KSIMD_DYN_FUNC_ATTR void kernel(
            const float* KSIMD_RESTRICT src,
                  float* KSIMD_RESTRICT dst,
            const size_t                size
        ) noexcept
        {
            namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
            using f32 = ns::op<float>;
            using batch_t = ns::Batch<float>;

            const batch_t c10 = f32::set(0.123);
            const batch_t c9  = f32::set(-0.456);
            const batch_t c8  = f32::set(0.789);
            const batch_t c7  = f32::set(-0.101);
            const batch_t c6  = f32::set(0.234);
            const batch_t c5  = f32::set(-0.567);
            const batch_t c4  = f32::set(0.890);
            const batch_t c3  = f32::set(-1.234);
            const batch_t c2  = f32::set(0.555);
            const batch_t c1  = f32::set(1.999);
            const batch_t c0  = f32::set(-0.777);

            const batch_t lower = f32::set(5.0);
            const batch_t upper = f32::set(10.0);

            auto compute_unit = [&](batch_t x) noexcept KSIMD_DYN_FUNC_ATTR
            {
                batch_t res = f32::mul_add(c10, x, c9);
                res = f32::mul_add(res, x, c8);
                res = f32::mul_add(res, x, c7);
                res = f32::mul_add(res, x, c6);
                res = f32::mul_add(res, x, c5);
                res = f32::mul_add(res, x, c4);
                res = f32::mul_add(res, x, c3);
                res = f32::mul_add(res, x, c2);
                res = f32::mul_add(res, x, c1);
                res = f32::mul_add(res, x, c0);

                res = f32::if_then_else(res < lower, lower, res);
                res = f32::if_then_else(res > upper, upper, res);
                res = ns::vmath::lerp(res, x, f32::set(0.5));

                return res;
            };

            size_t i = 0;
            const size_t step = f32::Lanes;

            // 主循环
            for (; i + step <= size; i += step)
            {
                batch_t x = f32::load(src + i);
                f32::store(dst + i, compute_unit(x));
            }

            // 尾处理
            if (const size_t tail = size - i; tail > 0)
            {
                batch_t x = f32::load_partial(src + i, tail);
                f32::store_partial(dst + i, compute_unit(x), tail);
            }
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    // 生成函数指针表
    KSIMD_DYN_DISPATCH_FUNC(kernel)

    // 封装外部接口函数
    void kernel(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        KSIMD_DYN_CALL(kernel)(src, dst, size);
    }
} // namespace MyNamespace

int main()
{
    constexpr size_t NUM = 1000003;

    // 使用对齐分配器
    std::vector<float, ksimd::AlignedAllocator<float>> src(NUM);
    std::vector<float, ksimd::AlignedAllocator<float>> dst(NUM);

    for (size_t i = 0; i < NUM; ++i)
    {
        src[i] = (float)i / NUM;
    }

    // 预热
    MyNamespace::kernel(src.data(), dst.data(), NUM);

    // 正式计时
    ScopeTimer timer("timer");
    for (int r = 0; r < 100; ++r)
    {
        MyNamespace::kernel(src.data(), dst.data(), NUM);
    }

    return 0;
}
#endif

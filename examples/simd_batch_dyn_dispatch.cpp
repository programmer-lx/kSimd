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
            using batch_t = ns::Batch<float>;

            const batch_t c10 = ns::set(0.123f);
            const batch_t c9  = ns::set(-0.456f);
            const batch_t c8  = ns::set(0.789f);
            const batch_t c7  = ns::set(-0.101f);
            const batch_t c6  = ns::set(0.234f);
            const batch_t c5  = ns::set(-0.567f);
            const batch_t c4  = ns::set(0.890f);
            const batch_t c3  = ns::set(-1.234f);
            const batch_t c2  = ns::set(0.555f);
            const batch_t c1  = ns::set(1.999f);
            const batch_t c0  = ns::set(-0.777f);

            const batch_t lower = ns::set(5.0f);
            const batch_t upper = ns::set(10.0f);

            auto compute_unit = [&](batch_t x) noexcept KSIMD_DYN_FUNC_ATTR
            {
                batch_t res = ns::mul_add(c10, x, c9);
                res = ns::mul_add(res, x, c8);
                res = ns::mul_add(res, x, c7);
                res = ns::mul_add(res, x, c6);
                res = ns::mul_add(res, x, c5);
                res = ns::mul_add(res, x, c4);
                res = ns::mul_add(res, x, c3);
                res = ns::mul_add(res, x, c2);
                res = ns::mul_add(res, x, c1);
                res = ns::mul_add(res, x, c0);

                res = ns::if_then_else(res < lower, lower, res);
                res = ns::if_then_else(res > upper, upper, res);
                res = ns::vmath::lerp(res, x, ns::set(0.5f));

                return res;
            };

            size_t i = 0;
            const size_t step = ns::Lanes<float>;

            // 主循环
            for (; i + step <= size; i += step)
            {
                batch_t x = ns::load(src + i);
                ns::store(dst + i, compute_unit(x));
            }

            // 尾处理
            if (const size_t tail = size - i; tail > 0)
            {
                batch_t x = ns::loadu_partial(src + i, tail);
                ns::storeu_partial(dst + i, compute_unit(x), tail);
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

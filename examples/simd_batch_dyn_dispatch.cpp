#include <string>
#include <vector>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_batch_dyn_dispatch.cpp"
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/aligned_allocate.hpp>
#include <kSimd/base_op.hpp>
#include <kSimd_extension/vmath.hpp>

#include "utils.hpp"

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
            using f64 = KSIMD_DYN_BASE_OP(double);

            // 预设 10 个系数，增加寄存器内的计算压力
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

                f64::batch_t res = f64::mul_add(c10, x, c9);
                res = f64::mul_add(res, x, c8);
                res = f64::mul_add(res, x, c7);
                res = f64::mul_add(res, x, c6);
                res = f64::mul_add(res, x, c5);
                res = f64::mul_add(res, x, c4);
                res = f64::mul_add(res, x, c3);
                res = f64::mul_add(res, x, c2);
                res = f64::mul_add(res, x, c1);
                res = f64::mul_add(res, x, c0);
                res = ext::vmath::clamp(res, c0, c1);
                res = ext::vmath::clamp(res, c2, c3);
                res = ext::vmath::clamp(res, c4, c5);

                f64::store(dst + i, res);
            }

            // 尾处理
            if (const size_t tail = size - i; tail > 0)
            {
                const f64::mask_t mask = f64::mask_from_lanes(static_cast<unsigned int>(tail));
                f64::batch_t x = f64::mask_load(src + i, mask);

                f64::batch_t res = f64::mul_add(c10, x, c9);
                res = f64::mul_add(res, x, c8);
                res = f64::mul_add(res, x, c7);
                res = f64::mul_add(res, x, c6);
                res = f64::mul_add(res, x, c5);
                res = f64::mul_add(res, x, c4);
                res = f64::mul_add(res, x, c3);
                res = f64::mul_add(res, x, c2);
                res = f64::mul_add(res, x, c1);
                res = f64::mul_add(res, x, c0);
                res = ext::vmath::clamp(res, c0, c1);
                res = ext::vmath::clamp(res, c2, c3);
                res = ext::vmath::clamp(res, c4, c5);

                f64::mask_store(dst + i, res, mask);
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
        const double* KSIMD_RESTRICT src,
              double* KSIMD_RESTRICT dst,
        const size_t                 size
    ) noexcept
    {
        KSIMD_DYN_CALL(kernel)(src, dst, size);
    }
} // namespace MyNamespace

int main()
{
    constexpr size_t NUM = 1000000;

    // 使用对齐分配器
    std::vector<double, ksimd::AlignedAllocator<double>> src(NUM);
    std::vector<double, ksimd::AlignedAllocator<double>> dst(NUM);

    for (size_t i = 0; i < NUM; ++i)
    {
        src[i] = (double)i / NUM;
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

#include <string>
#include <vector>
#include <iostream>

// 用户定义的宏，可以用于取消某条路径的分发，需要采用这种方式定义
// #undef KSIMD_DISABLE_AVX2_MAX
// #define KSIMD_DISABLE_AVX2_MAX

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
        bool flag = false;

        template<typename T>
        KSIMD_DYN_FUNC_ATTR void kernel_generic(
            const T* KSIMD_RESTRICT src,
                  T* KSIMD_RESTRICT dst,
            const size_t            size
        ) noexcept
        {
            if (!flag)
            {
                // 可查看当前所分发的路径是什么
                flag = true;
                std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
                std::cout << "current instruction: " << str << std::endl;
            }

            namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
            using batch_t = ns::Batch<T>;

            const batch_t c10 = ns::set(T{0.123});
            const batch_t c9  = ns::set(T{-0.456});
            const batch_t c8  = ns::set(T{0.789});
            const batch_t c7  = ns::set(T{-0.101});
            const batch_t c6  = ns::set(T{0.234});
            const batch_t c5  = ns::set(T{-0.567});
            const batch_t c4  = ns::set(T{0.890});
            const batch_t c3  = ns::set(T{-1.234});
            const batch_t c2  = ns::set(T{0.555});
            const batch_t c1  = ns::set(T{1.999});
            const batch_t c0  = ns::set(T{-0.777});

            const batch_t lower = ns::set(T{5.0});
            const batch_t upper = ns::set(T{10.0});

            auto compute_unit = [&](batch_t x) KSIMD_DYN_FUNC_ATTR
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
                res = ns::vmath::lerp(res, x, ns::set(T{0.5}));

                return res;
            };

            size_t i = 0;
            constexpr size_t step = ns::Lanes<T>;

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

        KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN
        void kernel_f32(
            const float* KSIMD_RESTRICT src,
                  float* KSIMD_RESTRICT dst,
            const size_t                size
        ) noexcept
        {
            kernel_generic(src, dst, size);
        }

        // KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN
        // void kernel_f64(
        //     const double* KSIMD_RESTRICT src,
        //           double* KSIMD_RESTRICT dst,
        //     const size_t                 size
        // ) noexcept
        // {
        //     kernel_generic(src, dst, size);
        // }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    // 生成函数指针表
    KSIMD_DYN_DISPATCH_FUNC(kernel_f32)
    // 封装外部接口函数
    void kernel(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        KSIMD_DYN_CALL(kernel_f32)(src, dst, size);
    }

    // KSIMD_DYN_DISPATCH_FUNC(kernel_f64)
    // void kernel(
    //     const double* KSIMD_RESTRICT src,
    //           double* KSIMD_RESTRICT dst,
    //     const size_t                size
    // ) noexcept
    // {
    //     KSIMD_DYN_CALL(kernel_f64)(src, dst, size);
    // }
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

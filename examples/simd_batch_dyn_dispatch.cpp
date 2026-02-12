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
        template<typename T, bool tag>
        KSIMD_DYN_FUNC_ATTR void kernel_template(
            const T* KSIMD_RESTRICT src,
                  T* KSIMD_RESTRICT dst,
            const size_t            size
        ) noexcept
        {
            // 可查看当前所分发的路径是什么
            std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
            std::cout << "current instruction: " << str << std::endl;

            if constexpr (tag)
            {
                std::cout << "has tag" << std::endl;
            }
            else
            {
                std::cout << "no tag" << std::endl;
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
        
        KSIMD_DYN_FUNC_ATTR void kernel_without_template() noexcept
        {
            std::cout << "without template." << std::endl;
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    // 生成函数指针表
    template<typename T, bool tag>
    KSIMD_DYN_DISPATCH_FUNC(kernel_template, <T, tag>);

    // 封装外部接口函数
    void kernel(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        // 直接call
        KSIMD_DYN_CALL(kernel_template, <float, false>)(src, dst, size);
    }

    void kernel_with_tag(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        // 先获取函数指针，再call
        auto fn = KSIMD_DYN_FUNC_POINTER(kernel_template, <float, true>);
        fn(src, dst, size);
    }
    
    // 非模板分发
    KSIMD_DYN_DISPATCH_FUNC(kernel_without_template);
    void kernel_without_template() noexcept
    {
        KSIMD_DYN_CALL(kernel_without_template)();
    }
} // namespace MyNamespace

int main()
{
    constexpr size_t NUM = 100;

    // 使用对齐分配器
    std::vector<float, ksimd::AlignedAllocator<float>> src(NUM);
    std::vector<float, ksimd::AlignedAllocator<float>> dst(NUM);

    for (size_t i = 0; i < NUM; ++i)
    {
        src[i] = (float)i / NUM;
    }

    // 模板分发
    MyNamespace::kernel(src.data(), dst.data(), NUM);
    MyNamespace::kernel_with_tag(src.data(), dst.data(), NUM);

    // 非模板分发
    MyNamespace::kernel_without_template();

    return 0;
}
#endif

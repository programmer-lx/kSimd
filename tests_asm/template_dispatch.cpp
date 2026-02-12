#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "template_dispatch.cpp"
#include <kSimd/core/dispatch_this_file.hpp>
#include <kSimd/core/dispatch_core.hpp>

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        template<typename T, bool tag>
        KSIMD_DYN_FUNC_ATTR void test_generic(
            const T* KSIMD_RESTRICT src,
                  T* KSIMD_RESTRICT dst,
            const size_t            size
        ) noexcept
        {
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
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace MyNamespace

#if KSIMD_ONCE
namespace MyNamespace
{
    // 生成函数指针表
    template<typename T, bool tag>
    KSIMD_DYN_DISPATCH_FUNC(test_generic, <T, tag>);

    // 封装外部接口函数
    void test(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        KSIMD_DYN_CALL(test_generic, <float, false>)(src, dst, size);
    }

    void test_with_tag(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        KSIMD_DYN_CALL(test_generic, <float, true>)(src, dst, size);
    }
} // namespace MyNamespace
#endif

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "template_dispatch.cpp"
#include <kSimd/core/dispatch_this_file.hpp>
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/dispatch_vmath.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    template<typename T, bool>
    KSIMD_DLL_LOCAL KSIMD_DYN_FUNC_ATTR void test_generic(
        const T* KSIMD_RESTRICT src,
              T* KSIMD_RESTRICT dst,
        const size_t            size
    ) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        ns::FullTag<T> t;
        using batch_t = ns::Batch<ns::FullTag<T>>;

        const batch_t c10 = ns::set(t, T{ 0.123 });
        const batch_t c9 = ns::set(t, T{ -0.456 });
        const batch_t c8 = ns::set(t, T{ 0.789 });
        const batch_t c7 = ns::set(t, T{ -0.101 });
        const batch_t c6 = ns::set(t, T{ 0.234 });
        const batch_t c5 = ns::set(t, T{ -0.567 });
        const batch_t c4 = ns::set(t, T{ 0.890 });
        const batch_t c3 = ns::set(t, T{ -1.234 });
        const batch_t c2 = ns::set(t, T{ 0.555 });
        const batch_t c1 = ns::set(t, T{ 1.999 });
        const batch_t c0 = ns::set(t, T{ -0.777 });

        const batch_t lower = ns::set(t, T{ 5.0 });
        const batch_t upper = ns::set(t, T{ 10.0 });

        size_t i = 0;
        const size_t step = ns::lanes(t);

        // 主循环
        for (; i + step <= size; i += step)
        {
            batch_t x = ns::load(t, src + i);

            batch_t res = ns::mul_add(t, c10, x, c9);
            res = ns::mul_add(t, res, x, c8);
            res = ns::mul_add(t, res, x, c7);
            res = ns::mul_add(t, res, x, c6);
            res = ns::mul_add(t, res, x, c5);
            res = ns::mul_add(t, res, x, c4);
            res = ns::mul_add(t, res, x, c3);
            res = ns::mul_add(t, res, x, c2);
            res = ns::mul_add(t, res, x, c1);
            res = ns::mul_add(t, res, x, c0);

            res = ns::if_then_else(t, ns::less(t, res, lower), lower, res);
            res = ns::if_then_else(t, ns::greater(t, res, lower), upper, res);
            res = ns::vmath::lerp(t, res, x, ns::set(t, T{ 0.5 }));

            ns::store(t, dst + i, res);
        }

        // 尾处理
        if (const size_t tail = size - i; tail > 0)
        {
            batch_t x = ns::loadu_partial(t, src + i, tail);

            batch_t res = ns::mul_add(t, c10, x, c9);
            res = ns::mul_add(t, res, x, c8);
            res = ns::mul_add(t, res, x, c7);
            res = ns::mul_add(t, res, x, c6);
            res = ns::mul_add(t, res, x, c5);
            res = ns::mul_add(t, res, x, c4);
            res = ns::mul_add(t, res, x, c3);
            res = ns::mul_add(t, res, x, c2);
            res = ns::mul_add(t, res, x, c1);
            res = ns::mul_add(t, res, x, c0);

            res = ns::if_then_else(t, ns::less(t, res, lower), lower, res);
            res = ns::if_then_else(t, ns::greater(t, res, lower), upper, res);
            res = ns::vmath::lerp(t, res, x, ns::set(t, T{ 0.5 }));

            ns::storeu_partial(t, dst + i, res, tail);
        }
    }
} // namespace KSIMD_DYN_INSTRUCTION

#if KSIMD_ONCE
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

// export C function
extern "C"
{
    void test_with_tag(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        KSIMD_DYN_CALL(test_generic, <float, true>)(src, dst, size);
    }
}
#endif

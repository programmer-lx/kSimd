#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "template_dispatch.cpp"
#include <kSimd/core/dispatch_this_file.hpp>
#include <kSimd/core/dispatch_core.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    template<typename T, bool tag>
    KSIMD_DLL_LOCAL KSIMD_DYN_FUNC_ATTR void test_generic(
        const T* KSIMD_RESTRICT src,
              T* KSIMD_RESTRICT dst,
        const size_t            size
    ) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        ns::Traits<T> t;
        
        using batch_t = ns::Batch<T>;
        const batch_t c  = ns::set(t, T{-0.777});

        const batch_t lower = ns::set(t, T{5.0});
        const batch_t upper = ns::set(t, T{10.0});

        size_t i = 0;
        const size_t step = ns::lanes(t);

        // 主循环
        for (; i + step <= size; i += step)
        {
            batch_t x = ns::load(t, src + i);

            batch_t res = ns::mul_add(t, c, x, c);

            ns::store(t, dst + i, res);
        }

        // 尾处理
        if (const size_t tail = size - i; tail > 0)
        {
            batch_t x = ns::loadu_partial(t, src + i, tail);

            batch_t res = ns::mul_add(t, c, x, c);

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

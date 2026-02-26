#include <string>
#include <vector>
#include <iostream>
#include <random>

// 可以用于取消某条路径的分发，需要采用这种方式定义
// 定义之前，必须取消定义，因为该文件会被重复包含，只#define，会造成重复定义
// #undef KSIMD_DISABLE_X86_V4
// #define KSIMD_DISABLE_X86_V4

// #undef KSIMD_DISABLE_X86_V3
// #define KSIMD_DISABLE_X86_V3

// 定义这个宏，然后包含 <kSimd/core/dispatch_this_file.hpp> 来让当前文件被递归包含，
// 每次包含，KSIMD_DYN_INSTRUCTION 宏会被重定义，在 KSIMD_DYN_INSTRUCTION 命名空间内的函数被复制多次，从而实现函数多态
// 之所以不使用模板参数来分发，是因为 __attribute__ 不能像模板参数那样多态，所以需要使用头文件递归包含的方式来模拟多态
// 在GCC / clang 下，KSIMD_DYN_FUNC_ATTR 可能是 __attribute__((target("avx2,fma,f16c"))) 或者其他属性，
// 只有添加 KSIMD_DYN_FUNC_ATTR 属性，这个函数才能被正确编译，否则可能会引起编译错误
#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "dyn_dispatch.cpp" // 递归包含当前文件
#include <kSimd/core/dispatch_this_file.hpp> // 由这个文件来驱动递归包含

// kSimd库文件，必须在 <kSimd/core/dispatch_this_file.hpp> 之后包含
#include <kSimd/core/aligned_allocate.hpp>
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/dispatch_vmath.hpp>

#include "utils.hpp"

namespace MyNamespace::KSIMD_DYN_INSTRUCTION
{
    // 必须添加 KSIMD_DYN_FUNC_ATTR
    // 被分发的函数可以是函数模板
    template<typename T, bool condition>
    KSIMD_DYN_FUNC_ATTR void kernel_template(const T* KSIMD_RESTRICT src, T* KSIMD_RESTRICT dst,
                                             const size_t size) noexcept
    {
        // 可查看当前所分发的路径是什么
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        std::cout << "current instruction: " << str << std::endl;

        if constexpr (condition)
        {
            std::cout << "true" << std::endl;
        }
        else
        {
            std::cout << "false" << std::endl;
        }

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

        auto compute_unit = [&](batch_t x) KSIMD_DYN_FUNC_ATTR
        {
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

            return res;
        };

        size_t i = 0;
        const size_t step = ns::lanes(t);

        // 主循环
        for (; i + step <= size; i += step)
        {
            batch_t x = ns::load(t, src + i);
            ns::store(t, dst + i, compute_unit(x));
        }

        // 尾处理
        if (const size_t tail = size - i; tail > 0)
        {
            batch_t x = ns::loadu_partial(t, src + i, tail);
            ns::storeu_partial(t, dst + i, compute_unit(x), tail);
        }
    }

    KSIMD_DYN_FUNC_ATTR int kernel_without_template() noexcept
    {
        std::cout << "without template." << std::endl;
        std::default_random_engine engine(std::random_device{}());
        std::uniform_int_distribution<> distribution(-100, 100);
        return distribution(engine); // 返回一个 [-100, 100] 的随机数
    }
} // namespace MyNamespace::KSIMD_DYN_INSTRUCTION

// 使用 KSIMD_ONCE 宏，让接下来的代码只被编译一次，不受文件递归包含的影响
#if KSIMD_ONCE
namespace MyNamespace // 命名空间一定要与上面的函数一致
{
    // 生成函数指针表
    // 上面的函数的模板参数是什么，这里就要填什么，要严格对应
    template<typename T, bool tag>
    KSIMD_DYN_DISPATCH_FUNC(kernel_template, <T, tag>); // 这里实际上是生成了一个函数指针数组

    // 封装外部接口函数 (如果这个函数要给外部使用，就不能是模板了)
    void kernel(
        const float* KSIMD_RESTRICT src,
              float* KSIMD_RESTRICT dst,
        const size_t                size
    ) noexcept
    {
        // 直接call
        // 必须先实例化模板，再call
        // call的时候，库会自动检测指令集的支持情况，自动选择最优指令集的函数版本进行调用
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
    // 被分发的函数可以有返回值
    KSIMD_DYN_DISPATCH_FUNC(kernel_without_template);
    int kernel_without_template() noexcept
    {
        return KSIMD_DYN_CALL(kernel_without_template)();
    }
} // namespace MyNamespace

int main()
{
    constexpr size_t NUM = 100;

    // 使用对齐分配器
    // 在X86架构下，128位向量需要16字节对齐，256位向量需要32字节对齐，如果不对齐，在调用 load 函数的时候，会出现运行时错误
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
    // 获取函数的返回值
    std::cout << MyNamespace::kernel_without_template() << std::endl;

    return 0;
}
#endif

#include <cassert>
#include <cmath>

#include <iostream>
#include <string>

#include <kSimd/simd_op.hpp>
#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_batch_dyn_dispatch.cpp" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch


#pragma message("dispatch target = " KSIMD_STR("" KSIMD_DYN_FUNC_ATTR))

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        KSIMD_DYN_FUNC_ATTR void kernel_dyn_impl(
            const float* arr,
            const float* arr2,
            const float* arr3,
            const size_t N,
            float* out_result) noexcept
        {
            using op = KSIMD_DYN_SIMD_OP(float);
            using trait = op::traits;
            using batch_t = trait::batch_t;
            constexpr size_t Step = trait::Lanes;

            // 测试
            std::string cur_intrinsic = KSIMD_STR("" KSIMD_DYN_FUNC_ATTR);
#if defined(KSIMD_COMPILER_MSVC)
            assert(cur_intrinsic == "\"\"");
#elif defined(KSIMD_COMPILER_GCC) || defined(KSIMD_COMPILER_CLANG)
            assert(cur_intrinsic.contains("__attribute__((target("));
#else
    #error "Unknown compiler."
#endif

            size_t i = 0;
            for (; i + Step <= N; i += Step)
            {
                batch_t a = op::loadu(arr + i);
                batch_t b = op::loadu(arr2 + i);
                batch_t c = op::loadu(arr3 + i);

                batch_t tmp = op::mul_add(a, b, c);
                op::storeu(out_result + i, tmp);
            }
            for (; i < N; ++i)
            {
                out_result[i] = arr[i] * arr2[i] + arr3[i];
            }
        }
    }
}


#if KSIMD_ONCE
// declare a function table
namespace MyNamespace
{
    KSIMD_DYN_DISPATCH_FUNC(kernel_dyn_impl);
    void kernel(const float* arr, const float* arr2, const float* arr3, const size_t N, float* out_result) noexcept
    {
        KSIMD_DYN_CALL(kernel_dyn_impl)(arr, arr2, arr3, N, out_result);
    }
}
#endif

#if KSIMD_ONCE
int main()
{
    float numbers[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float numbers2[] = { -1, -2, -3, -4, -5, -6, -7, -8, -9 };
    float numbers3[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };

    float result[8] = {};
    MyNamespace::kernel(numbers, numbers2, numbers3, 8, result);

    for (int i = 0; i < 8; ++i)
    {
        float expected = numbers[i] * numbers2[i] + numbers3[i];
        assert(std::abs(expected - result[i]) <= 1e-5f);
    }

    std::cout << "SUCCEED" << std::endl;

    return 0;
}
#endif
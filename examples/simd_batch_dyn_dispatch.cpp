#include <cassert>
#include <cmath>

#include <iostream>
#include <string>
#include <vector>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_batch_dyn_dispatch.cpp" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

#include <kSimd/aligned_allocate.hpp>
#include <kSimd/simd_op.hpp>
#include "utils.hpp"


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
            using batch_t = op::batch_t;
            constexpr size_t Step = op::Lanes;

            // 测试
            [[maybe_unused]] std::string cur_intrinsic = KSIMD_STR("" KSIMD_DYN_FUNC_ATTR);
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

        KSIMD_DYN_FUNC_ATTR void speed_test_impl(
            const float* KSIMD_RESTRICT a,
            const float* KSIMD_RESTRICT b,
                  float* KSIMD_RESTRICT out,
            const size_t                size
        ) noexcept
        {
            ScopeTimer timer("div: " KSIMD_STR(KSIMD_DYN_INSTRUCTION));

            using op = KSIMD_DYN_SIMD_OP(float);
            constexpr auto Lanes = op::Lanes;
            using batch_t = op::batch_t;

            size_t i = 0;
            float x = random_f(-100.0f, 100.0f);
            for (; i + Lanes <= size; i += Lanes)
            {
                batch_t data = op::div(op::load(a + i), op::load(b + i));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::mul_add(data, op::set(x), op::set(x));
                data = op::min(data, op::set(100));
                data = op::max(data, op::set(-100));
                op::store(out + i, data);
            }
            for (; i < size; ++i)
            {
                out[i] = a[i] / b[i];
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

    KSIMD_DYN_DISPATCH_FUNC(speed_test_impl);
    void speed_test(
            const float* KSIMD_RESTRICT a,
            const float* KSIMD_RESTRICT b,
                  float* KSIMD_RESTRICT out,
            const size_t                size
        ) noexcept
    {
        // KSIMD_DYN_CALL(speed_test_div_impl)(a, b, out, size);
        for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(speed_test_impl)); ++i)
        {
            KSIMD_DETAIL_PFN_TABLE_FULL_NAME(speed_test_impl)[i](a, b, out, size);
        }
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
        [[maybe_unused]] float expected = numbers[i] * numbers2[i] + numbers3[i];
        assert(std::abs(expected - result[i]) <= 1e-5f);
    }

    // div test
    std::vector<float, ksimd::AlignedAllocator<float>> a;
    std::vector<float, ksimd::AlignedAllocator<float>> b;
    std::vector<float, ksimd::AlignedAllocator<float>> c;
    for (unsigned long long i = 0; i < 2048ull * 2048ull * 12ull; ++i)
    {
        a.push_back(random_f(-100.0f, 100.0f));
        b.push_back(random_f(-100.0f, 100.0f));
        c.push_back(-1.0f);
    }
    MyNamespace::speed_test(a.data(), b.data(), c.data(), c.size());
    [[maybe_unused]] volatile void* ptr = &c;

    std::cout << "SUCCEED" << std::endl;

    return 0;
}
#endif
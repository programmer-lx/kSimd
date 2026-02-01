#include <iostream>
#include <string>
#include <vector>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_batch_dyn_dispatch.cpp" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

#include <kSimd/aligned_allocate.hpp>
#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>

#include "utils.hpp"


#pragma message("dispatch target = " KSIMD_STR("" KSIMD_DYN_FUNC_ATTR))

namespace MyNamespace
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        KSIMD_DYN_FUNC_ATTR
        void add_and_clamp(
            const double* KSIMD_RESTRICT arr,
            const double                 min,
            const double                 max,
                  double* KSIMD_RESTRICT out,
            const size_t                 size
        ) noexcept
        {
            KSIMD_ENABLE_OPERATORS();
            namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;
            using f64 = KSIMD_DYN_OP(double);

            f64::batch_t min_val = f64::set(min);
            f64::batch_t max_val = f64::set(max);

            size_t i = 0;
            for (; i + f64::Lanes <= size; i += f64::Lanes)
            {
                f64::batch_t data = f64::load(arr + i);
                data += f64::set(1000);
                data = ext::math::clamp(data, min_val, max_val);
                f64::store(out + i, data);
            }

            // 尾处理，处理剩余的长度不足Lanes的元素
            // 使用 mask_load | mask_store 加载和存储部分元素，而非全量加载和存储
            const size_t tail = size - i;
            const f64::mask_t mask = f64::mask_from_lanes(static_cast<unsigned int>(tail));
            f64::batch_t tail_data = f64::mask_load(arr + i, mask);
            tail_data = f64::add(tail_data, f64::set(1000));
            tail_data = ext::math::clamp(tail_data, min_val, max_val);
            f64::mask_store(out + i, tail_data, mask);
        }
    }
}

#if KSIMD_ONCE
namespace MyNamespace
{
    KSIMD_DYN_DISPATCH_FUNC(add_and_clamp)
    void add_and_clamp(
        const double* KSIMD_RESTRICT arr,
        const double                 min,
        const double                 max,
              double* KSIMD_RESTRICT out,
        const size_t                 size
    ) noexcept
    {
        KSIMD_DYN_CALL(add_and_clamp)(arr, min, max, out, size);
    }
}
#endif

#if KSIMD_ONCE
int main()
{
    constexpr size_t NUM = 3211;

    std::vector<double, ksimd::AlignedAllocator<double>> arr(NUM);
    for (size_t i = 0; i < NUM; ++i)
    {
        arr[i] = random_f(-100.0f, 100.0f);
    }

    std::vector<double, ksimd::AlignedAllocator<double>> result(NUM);
    MyNamespace::add_and_clamp(arr.data(), -10000, 10000, result.data(), NUM);

    bool succeed = true;
    for (size_t i = 0; i < NUM; ++i)
    {
        const auto& r = result[i];

        if (!( (r >= -10000 && r <= 10000) || (r != arr[i] + 1000) ))
        {
            succeed = false;
        }
    }

    if (succeed)
        std::cout << "SUCCEED" << std::endl;
    else
        std::cout << "FAILED" << std::endl;

    return 0;
}
#endif
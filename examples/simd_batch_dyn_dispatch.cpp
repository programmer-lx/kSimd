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
            const float* KSIMD_RESTRICT arr,
            const float                 min,
            const float                 max,
                  float* KSIMD_RESTRICT out,
            const size_t                size
        ) noexcept
        {
            namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;
            using f32 = KSIMD_DYN_SIMD_OP(float);

            f32::batch_t min_val = f32::set(min);
            f32::batch_t max_val = f32::set(max);

            size_t i = 0;
            for (; i + f32::Lanes <= size; i += f32::Lanes)
            {
                f32::batch_t data = f32::load(arr + i);
                data = f32::add(data, f32::set(1000));
                data = ext::math::clamp(data, min_val, max_val);
                f32::store(out + i, data);
            }

            // 尾处理，处理剩余的长度不足Lanes的元素
            // 使用 mask_load | mask_store 加载和存储部分元素，而非全量加载和存储
            const size_t tail = size - i;
            const f32::mask_t mask = f32::mask_from_lanes(static_cast<unsigned int>(tail));
            f32::batch_t tail_data = f32::mask_load(arr + i, mask);
            tail_data = f32::add(tail_data, f32::set(1000));
            tail_data = ext::math::clamp(tail_data, min_val, max_val);
            f32::mask_store(out + i, tail_data, mask);
        }
    }
}

#if KSIMD_ONCE
namespace MyNamespace
{
    KSIMD_DYN_DISPATCH_FUNC(add_and_clamp)
    void add_and_clamp(
        const float* KSIMD_RESTRICT arr,
        const float                 min,
        const float                 max,
              float* KSIMD_RESTRICT out,
        const size_t                size
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

    std::vector<float, ksimd::AlignedAllocator<float>> arr(NUM);
    for (size_t i = 0; i < NUM; ++i)
    {
        arr[i] = random_f(-100.0f, 100.0f);
    }

    std::vector<float, ksimd::AlignedAllocator<float>> result(NUM);
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
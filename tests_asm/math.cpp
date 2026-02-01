#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "math.cpp" // this file
#include <kSimd/dispatch_this_file.hpp>
#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void kernel_impl(
        const float* KSIMD_RESTRICT v,
        const float* KSIMD_RESTRICT min,
        const float* KSIMD_RESTRICT max,
              float* KSIMD_RESTRICT out,
        const size_t                size
    ) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(float);
        namespace math = ksimd::ext::KSIMD_DYN_INSTRUCTION::math;
        constexpr size_t Lanes = op::Lanes;

        for (size_t i = 0; i + Lanes <= size; i += Lanes)
        {
            op::batch_t data = math::clamp(op::load(v + i), op::load(min + i), op::load(max + i));
            data = math::lerp(data, op::add(data, op::set(5)), op::set(0.5f));
            data = math::safe_clamp(data, op::set(1), op::set(6));
            data = math::sin(data);
            op::store(out + i, data);
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_impl);
void kernel(
    const float* v,
    const float* min,
    const float* max,
          float* out,
          size_t size
) noexcept
{
    KSIMD_DYN_CALL(kernel_impl)(v, min, max, out, size);
}
#endif
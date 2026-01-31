
#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "clamp.cpp" // this file
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

        size_t i = 0;
        for (; i + Lanes <= size; i += Lanes)
        {
            op::store(out + i, math::clamp(op::load(v + i), op::load(min + i), op::load(max + i)));
        }
        for (; i < size; ++i)
        {
            out[i] = math::clamp(v[i], min[i], max[i]);
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
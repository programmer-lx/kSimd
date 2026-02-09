#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "operator_overload.cpp" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/vmathf.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void kernel(
        float* KSIMD_RESTRICT out_ptr,
        const float* KSIMD_RESTRICT in_ptr,
        float scalar_val
    ) noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<float>;
        using batch_t = ns::Batch<float>;

        batch_t v0 = op::load(in_ptr + 0 * op::Lanes);
        batch_t v1 = op::load(in_ptr + 1 * op::Lanes);
        batch_t v2 = op::set(scalar_val);
        batch_t v3 = v0 + v1;             // operator+ (Batch, Batch)
        batch_t v4 = v0 - v2;             // operator- (Batch, Batch)
        batch_t v5 = v1 * float(2.5);    // operator* (Batch, Scalar)
        batch_t v6 = v3 / (v2 + float(1.0)); // operator/ (Batch, Batch) + static_assert check
        batch_t v7 = op::zero();
        batch_t v8 = op::set(float(0.123));
        batch_t v9 = v4 * v5;

        v0 += v9;                         // operator+= (Batch, Batch)
        v1 *= scalar_val;                 // operator*= (Batch, Scalar)
        v7 -= v6;                         // operator-= (Batch, Batch)

        v8 = op::mul_add(v0, v1, v2);
        v9 = op::mul_add(v3, v4, v5);

        auto mask = v8 > v9;              // operator> (Batch, Batch) -> Mask
        auto n_mask = ~mask;              // operator~ (Mask)
    
        // lerp(a, b, t) -> a + t * (b - a)
        batch_t res_a = ns::vmathf::lerp(v0, v1, v8);
        batch_t res_b = ns::vmathf::lerp(v2, v3, v9);

        batch_t final_v1 = op::if_then_else(mask, res_a, v7);
        batch_t final_v2 = op::if_then_else(n_mask, res_b, v5);
    
        batch_t final_res = (final_v1 + final_v2) * v8;

        op::store(out_ptr, final_res);
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel)
void kernel(
    float* KSIMD_RESTRICT out_ptr,
    const float* KSIMD_RESTRICT in_ptr,
    float scalar_val
) noexcept
{
    KSIMD_DYN_CALL(kernel)(out_ptr, in_ptr, scalar_val);
}
#endif

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "bitcast.cpp" // this file
#include <kSimd/dispatch_this_file.hpp>
#include <kSimd/base_op.hpp>
#include <kSimd/type_op.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void kernel_impl() noexcept
    {
        using f32 = KSIMD_DYN_BASE_OP(float);
        using f64 = KSIMD_DYN_BASE_OP(double);
        using type = KSIMD_DYN_TYPE_OP();
        
        // -------------------- self -> self --------------------
        // f32 -> f32
        {
            f32::batch_t a = f32::set(5);
            f32::batch_t b = type::bit_cast<f32::batch_t>(a);
            [[maybe_unused]] volatile void* ptr = &b;
        }
        // f64 -> f64
        {
            f64::batch_t a = f64::set(5);
            f64::batch_t b = type::bit_cast<f64::batch_t>(a);
            [[maybe_unused]] volatile void* ptr = &b;
        }

        // -------------------- f32 -> ? --------------------
        // f32 -> f64
        {
            f32::batch_t a = f32::set(6);
            f64::batch_t b = type::bit_cast<f64::batch_t>(a);
            [[maybe_unused]] volatile void* ptr = &b;
        }

        // -------------------- f64 -> ? --------------------
        // f64 -> f32
        {
            f64::batch_t a = f64::set(5);
            f32::batch_t b = type::bit_cast<f32::batch_t>(a);
            [[maybe_unused]] volatile void* ptr = &b;
        }

        // -------------------- i32 -> ? --------------------

        // -------------------- u32 -> ? --------------------
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_impl);
void kernel() noexcept
{
    KSIMD_DYN_CALL(kernel_impl)();
}
#endif
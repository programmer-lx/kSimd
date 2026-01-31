#include "../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "type_op/bitcast_all.cpp" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/simd_op.hpp>
#include <kSimd/type_op.hpp>

using namespace ksimd;

// #define KSIMD_DYN_INSTRUCTION AVX
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR void kernel() noexcept
    {
        using f32_op = KSIMD_DYN_SIMD_OP(float32);
        using f32 = f32_op::traits;

        using f64_op = KSIMD_DYN_SIMD_OP(float64);
        using f64 = f64_op::traits;

        // using i32 = KSIMD_DYN_SIMD_TRAITS(int32_t);
        // using i32_op = KSIMD_DYN_SIMD_OP(int32_t);
        //
        // using u32 = KSIMD_DYN_SIMD_TRAITS(uint32_t);
        // using u32_op = KSIMD_DYN_SIMD_OP(uint32_t);

        using type_op = KSIMD_DYN_TYPE_OP();

        // -------------------- self -> self --------------------
        // f32 -> f32
        {
            f32::batch_t a = f32_op::set(5);
            f32::batch_t b = type_op::bit_cast<f32::batch_t>(a);
            EXPECT_TRUE(simd_type_bit_equal(a, b));
        }
        // f64 -> f64
        {
            f64::batch_t a = f64_op::set(5);
            f64::batch_t b = type_op::bit_cast<f64::batch_t>(a);
            EXPECT_TRUE(simd_type_bit_equal(a, b));
        }

        // -------------------- f32 -> ? --------------------
        // f32 -> f64
        {
            f32::batch_t a = f32_op::set(6);
            f64::batch_t b = type_op::bit_cast<f64::batch_t>(a);
            EXPECT_TRUE(simd_type_bit_equal(a, b));
        }

        // -------------------- f64 -> ? --------------------
        // f64 -> f32
        {
            f64::batch_t a = f64_op::set(5);
            f32::batch_t b = type_op::bit_cast<f32::batch_t>(a);
            EXPECT_TRUE(simd_type_bit_equal(a, b));
        }

        // -------------------- i32 -> ? --------------------

        // -------------------- u32 -> ? --------------------
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel);
TEST(bit_cast, all)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel)); ++idx)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel)[idx]();
    }
}

struct doublex1
{
    alignas(16) double v[1];
};

TEST(bit_cast, check_alignment)
{
    constexpr auto align = alignof(doublex1) - offsetof(doublex1, v);
    EXPECT_TRUE(align == 16);
}
#endif


// main function
#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
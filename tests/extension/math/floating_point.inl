#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/math/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

// 下面这两个文件一定要放在dispatch_this_file.hpp之后
#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>


// ------------------------------------------ sin ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sin(FLOAT_T x, FLOAT_T* KSIMD_RESTRICT out) noexcept
    {
        using traits = KSIMD_DYN_SIMD_OP(FLOAT_T)::traits;
        using op = KSIMD_DYN_SIMD_OP(FLOAT_T);
        constexpr size_t Step = traits::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, ext::math::sin(op::set(x)));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(sin);

TEST(dyn_dispatch_FLOAT_T, sin)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sin)); ++idx)
    {
        FLOAT_T n = FLOAT_C(-10.0);
        for (int c = 0; c < 100; ++c)
        {
            alignas(ALIGNMENT) FLOAT_T out[TOTAL];

            for (size_t i = 0; i < TOTAL; ++i) out[i] = FLOAT_C(-100.0);

            KSIMD_DETAIL_PFN_TABLE_FULL_NAME(sin)[idx](n, out);

            for (size_t i = 0; i < TOTAL; ++i)
            {
                EXPECT_NEAR(out[i], std::sin(n), FLOAT_T_EPSILON);
            }

            n += 0.002;
        }
    }
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
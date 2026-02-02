#include "../../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/math/FLOAT_T/floating_point.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch

// 下面这两个文件一定要放在dispatch_this_file.hpp之后
#include <kSimd/simd_op.hpp>
#include <kSimd_extension/math.hpp>


// ------------------------------------------ sin ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void sin() noexcept
    {
        using op = KSIMD_DYN_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        alignas(ALIGNMENT) FLOAT_T in_data[Lanes];
        alignas(ALIGNMENT) FLOAT_T out_data[Lanes];

        auto verify = [&](std::function<FLOAT_T(size_t)> generator) {
            for (size_t i = 0; i < Lanes; ++i) in_data[i] = generator(i);
            op::store(out_data, ext::math::sin(op::load(in_data)));
            for (size_t i = 0; i < Lanes; ++i) {
                if (std::isnan(in_data[i]) || std::isinf(in_data[i])) {
                    EXPECT_TRUE(std::isnan(out_data[i])); // sin(inf) 或 sin(nan) 均为 NaN
                } else {
                    EXPECT_NEAR(out_data[i], std::sin(in_data[i]), std::numeric_limits<FLOAT_T>::epsilon() * 100);
                }
            }
        };

        // 1. 常规步进测试 (-10.0 to 10.0)
        for (FLOAT_T val = FLOAT_T(-10); val < FLOAT_T(10); val += FLOAT_T(0.5)) {
            verify([&](size_t i) { return val + FLOAT_T(i) * FLOAT_T(0.01); });
        }

        // 2. 特殊值测试
        verify([](size_t i) {
            const FLOAT_T cases[] = { FLOAT_T(0), FLOAT_T(-0.0), qNaN<FLOAT_T>, inf<FLOAT_T>, -inf<FLOAT_T> };
            return cases[i % 5];
        });
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(sin)
#endif

// ------------------------------------------ lerp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void lerp() noexcept
    {
        using op = KSIMD_DYN_OP(FLOAT_T);
        constexpr size_t Lanes = op::Lanes;
        namespace ext = ksimd::ext::KSIMD_DYN_INSTRUCTION;

        alignas(ALIGNMENT) FLOAT_T aa[Lanes], bb[Lanes], tt[Lanes], rr[Lanes];

        auto verify_lerp = [&](FLOAT_T va, FLOAT_T vb, FLOAT_T vt, std::optional<FLOAT_T> expected = std::nullopt) {
            for (size_t i = 0; i < Lanes; ++i) { aa[i] = va; bb[i] = vb; tt[i] = vt; }
            op::store(rr, ext::math::lerp(op::load(aa), op::load(bb), op::load(tt)));
            
            for (size_t i = 0; i < Lanes; ++i) {
                if (std::isnan(va) || std::isnan(vb) || std::isnan(vt)) {
                    EXPECT_TRUE(std::isnan(rr[i]));
                } else if (expected) {
                    EXPECT_NEAR(rr[i], *expected, std::numeric_limits<FLOAT_T>::epsilon() * 10);
                }
            }
        };

        // 1. 基础插值 (t=0.6, a=0, b=2 -> 1.2)
        verify_lerp(FLOAT_T(0), FLOAT_T(2), FLOAT_T(0.6), FLOAT_T(1.2));

        // 2. 边界 t=0 和 t=1
        verify_lerp(FLOAT_T(10), FLOAT_T(20), FLOAT_T(0), FLOAT_T(10));
        verify_lerp(FLOAT_T(10), FLOAT_T(20), FLOAT_T(1), FLOAT_T(20));

        // 3. 外插 (t > 1 或 t < 0)
        verify_lerp(FLOAT_T(0), FLOAT_T(2), FLOAT_T(2), FLOAT_T(4));
        verify_lerp(FLOAT_T(0), FLOAT_T(2), FLOAT_T(-1), FLOAT_T(-2));

        // 4. 特殊值测试 (NaN & Inf)
        verify_lerp(qNaN<FLOAT_T>, FLOAT_T(1), FLOAT_T(0.5)); // 结果应为 NaN
        verify_lerp(FLOAT_T(0), inf<FLOAT_T>, FLOAT_T(0.5), inf<FLOAT_T>); 
        verify_lerp(FLOAT_T(0), FLOAT_T(1), inf<FLOAT_T>, inf<FLOAT_T>);
    }
}

#if KSIMD_ONCE
TEST_ONCE_DYN(lerp)
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
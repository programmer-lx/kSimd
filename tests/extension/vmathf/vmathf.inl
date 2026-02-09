#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/vmathf/vmathf.inl" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/vmathf.hpp>

// ------------------------------------------ lerp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    KSIMD_DYN_FUNC_ATTR
    void lerp() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        using op = ns::op<FLOAT_T>;
        using batch_t = ns::Batch<FLOAT_T>;
        
        constexpr size_t Lanes = op::Lanes;
        alignas(op::Alignment) FLOAT_T act[Lanes], exp[Lanes];

        #define check(actual, expected, msg) \
        do { \
            op::store(act, actual); \
            op::store(exp, expected); \
            for (size_t i = 0; i < Lanes; ++i) \
                EXPECT_TRUE(approximately(act[i], exp[i], FLOAT_T(std::numeric_limits<FLOAT_T>::epsilon() * 10))) \
                    << "Forwarding failed: " << msg << " at lane " << i; \
        } while (0)

        // 准备测试数据
        batch_t a = op::set(FLOAT_T(10.0)); // 起点
        batch_t b = op::set(FLOAT_T(20.0)); // 终点
        batch_t t = op::set(FLOAT_T(0.5));  // 插值系数
        check(ns::vmathf::lerp(a, b, t), op::set(15.0), "lerp(Batch, Batch, Batch)");

        t = op::set(FLOAT_T(-0.3));
        check(ns::vmathf::lerp(a, b, t), op::set(FLOAT_T(7)), "");

        #undef check
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(lerp)
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
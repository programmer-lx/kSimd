#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "extension/vmath/vmath.inl" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/extension/dispatch_vmath.hpp>

// ------------------------------------------ lerp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {

    static void perform_lerp_check(
        const FLOAT_T* act,
        const FLOAT_T* exp,
        size_t lanes,
        const char* msg)
    {
        for (size_t i = 0; i < lanes; ++i) {
            bool ok = false;
            if (std::isnan(exp[i])) {
                ok = std::isnan(act[i]);
            } else {
                ok = approximately(act[i], exp[i], FLOAT_T(std::numeric_limits<FLOAT_T>::epsilon() * 10));
            }
            EXPECT_TRUE(ok) << "Test failed: " << msg << " at lane " << i
                            << " [Actual: " << act[i] << ", Expected: " << exp[i] << "]";
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void lerp() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        using batch_t = ns::Batch<FLOAT_T>;

        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        // 显式对齐的本地缓冲区
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T act_buf[Lanes];
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T exp_buf[Lanes];

        #define CHECK(actual, expected, msg) \
        do { \
            ns::store(act_buf, actual); \
            ns::store(exp_buf, expected); \
            perform_lerp_check(act_buf, exp_buf, Lanes, msg); \
        } while (0)

        const batch_t a = ns::set(FLOAT_T(10.0));
        const batch_t b = ns::set(FLOAT_T(20.0));

        // --- 1. 标准范围内插值 ---
        CHECK(ns::vmath::lerp(a, b, ns::set(FLOAT_T(0.0))), a, "t=0");
        CHECK(ns::vmath::lerp(a, b, ns::set(FLOAT_T(1.0))), b, "t=1");
        CHECK(ns::vmath::lerp(a, b, ns::set(FLOAT_T(0.5))), ns::set(FLOAT_T(15.0)), "t=0.5");

        // --- 2. 外插测试 (Extrapolation) ---
        // t = -0.3  => 10 + (-0.3 * 10) = 7
        CHECK(ns::vmath::lerp(a, b, ns::set(FLOAT_T(-0.3))), ns::set(FLOAT_T(7.0)), "t < 0");
        // t = 1.2   => 10 + (1.2 * 10) = 22
        CHECK(ns::vmath::lerp(a, b, ns::set(FLOAT_T(1.2))),  ns::set(FLOAT_T(22.0)), "t > 1");

        // --- 3. 负数范围与反向插值 ---
        // a=10, b=-10, t=0.2 => 10 + 0.2*(-20) = 6
        CHECK(ns::vmath::lerp(a, ns::set(FLOAT_T(-10.0)), ns::set(FLOAT_T(0.2))), ns::set(FLOAT_T(6.0)), "mixed sign");

        // --- 4. 极端值测试 ---
        // 同值插值
        CHECK(ns::vmath::lerp(a, a, ns::set(FLOAT_T(0.7))), a, "a == b");

        // 零区间插值 (a=0, b=0)
        batch_t zero = ns::set(FLOAT_T(0.0));
        CHECK(ns::vmath::lerp(zero, zero, ns::set(FLOAT_T(0.5))), zero, "zero range");

        // NaN 传播测试 (基于你对 clamp 的 NaN 处理逻辑，lerp 也应遵循)
        if constexpr (std::numeric_limits<FLOAT_T>::has_quiet_NaN) {
            batch_t nan_v = ns::set(std::numeric_limits<FLOAT_T>::quiet_NaN());
            // 如果 t 是 NaN，结果应当是 NaN
            CHECK(ns::vmath::lerp(a, b, nan_v), nan_v, "NaN propagation in t");
        }

        #undef CHECK
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(lerp)
#endif

// ------------------------------------------ clamp ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION {
    static void perform_clamp_check(
        const FLOAT_T* act, 
        const FLOAT_T* exp, 
        size_t lanes, 
        const char* msg) 
    {
        for (size_t i = 0; i < lanes; ++i) {
            bool ok = false;
            // 处理 NaN 的比较：std::isnan 不能直接用 == 比较
            if (std::isnan(exp[i])) {
                ok = std::isnan(act[i]);
            } else {
                ok = (act[i] == exp[i]); // Clamp 通常应精确等于边界或原值
            }
            
            EXPECT_TRUE(ok) << "Clamp failed: " << msg << " at lane " << i 
                            << " [Actual: " << act[i] << ", Expected: " << exp[i] << "]";
        }
    }

    KSIMD_DYN_FUNC_ATTR
    void clamp_test() noexcept {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
        
        using batch_t = ns::Batch<FLOAT_T>;
        using Opt = ksimd::FloatMinMaxOption;
        
        constexpr size_t Lanes = ns::Lanes<FLOAT_T>;
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T act_buf[Lanes];
        alignas(ns::Alignment<FLOAT_T>) FLOAT_T exp_buf[Lanes];

        // 辅助宏，仅用于 store 和调用静态检查函数
        #define CLAMP_CHECK(actual, expected, msg) \
        do { \
            ns::store(act_buf, actual); \
            ns::store(exp_buf, expected); \
            perform_clamp_check(act_buf, exp_buf, Lanes, msg); \
        } while (0)

        const batch_t v_mid = ns::set(FLOAT_T(15.0));
        const batch_t v_low = ns::set(FLOAT_T(5.0));
        const batch_t v_high = ns::set(FLOAT_T(25.0));
        const batch_t min_v = ns::set(FLOAT_T(10.0));
        const batch_t max_v = ns::set(FLOAT_T(20.0));
        const batch_t nan_v = ns::set(ksimd::QNaN<FLOAT_T>);

        // ==========================================================
        // 1. 常规测试：min, max 均不是 NaN
        // ==========================================================
        
        // 测试原值在范围内
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_mid, min_v, max_v), v_mid, "Native: within range");
        // 测试下限触发
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_low, min_v, max_v), min_v, "Native: lower bound");
        // 测试上限触发
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_high, min_v, max_v), max_v, "Native: upper bound");
        // 测试 v 是 NaN 的传播 (利用你提到的 v 在右侧特性)
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(nan_v, min_v, max_v), nan_v, "Native: v is NaN propagation");

        // ==========================================================
        // 2. 异常测试：min, max 存在 NaN 的三种情况
        // ==========================================================

        // 情况 A: min 是 NaN, max 正常
        // 逻辑：max(NaN, min(20, 15)) -> max(NaN, 15) -> 返回 15 (Native 下硬件通常返回第二个参数)
        // 注意：如果你需要针对这种情况也传播 NaN，则必须使用 CheckNaN 模式
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_mid, nan_v, max_v), v_mid, "Native: min is NaN (returns v)");
        CLAMP_CHECK(ns::vmath::clamp<Opt::CheckNaN>(v_mid, nan_v, max_v), nan_v, "CheckNaN: min is NaN (returns NaN)");

        // 情况 B: max 是 NaN, min 正常
        // 逻辑：max(10, min(NaN, 15)) -> max(10, 15) -> 15
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_mid, min_v, nan_v), v_mid, "Native: max is NaN (returns v)");
        CLAMP_CHECK(ns::vmath::clamp<Opt::CheckNaN>(v_mid, min_v, nan_v), nan_v, "CheckNaN: max is NaN (returns NaN)");

        // 情况 C: min, max 全是 NaN
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(v_mid, nan_v, nan_v), v_mid, "Native: both min/max NaN");
        CLAMP_CHECK(ns::vmath::clamp<Opt::CheckNaN>(v_mid, nan_v, nan_v), nan_v, "CheckNaN: both min/max NaN");

        // ==========================================================
        // 3. 边界混合测试 (v 也是 NaN)
        // ==========================================================
        // 无论何种情况，只要 v 是 NaN，由于它在最右侧，Native 模式也应返回 NaN
        CLAMP_CHECK(ns::vmath::clamp<Opt::Native>(nan_v, nan_v, max_v), nan_v, "Native: min and v are NaN");

        #undef CLAMP_CHECK
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(clamp_test)
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
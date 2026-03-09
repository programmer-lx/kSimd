#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "base_op/SIGNED_T/signed.inl" // this file
#include <kSimd/core/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/core/dispatch_core.hpp>
#include <kSimd/core/aligned_allocate.hpp>
#include <vector>

// ------------------------------------------ abs ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void abs() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;
        
        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> test(Lanes);

        // 基础数值：正数与负数
        ns::store(t, test.data(), ns::abs(t, ns::set(t, TYPE_T(-5))));
        EXPECT_TRUE(array_equal(test.data(), Lanes, TYPE_T(5)));

        #if KSIMD_TEST_FP
        {
            // -0.0 -> 0.0 (符号位必须清除)
            ns::store(t, test.data(), ns::abs(t, ns::set(t, TYPE_T(-0.0))));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_EQ(test[i], TYPE_T(0.0));
                EXPECT_FALSE(ksimd::sign_bit(test[i]));
            }

            // -Inf -> Inf
            ns::store(t, test.data(), ns::abs(t, ns::set(t, -ksimd::Inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_inf(test[i]) && test[i] > 0);
            }
        }
        #endif
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(abs)
#endif

// ------------------------------------------ neg ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void neg() noexcept
    {
        namespace ns = ksimd::KSIMD_DYN_INSTRUCTION; TAG_T t;

        const size_t Lanes = ns::lanes(t);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> src(Lanes);
        std::vector<TYPE_T, ksimd::AlignedAllocator<TYPE_T>> dst(Lanes);

        // 1. 基础数值测试：正变负，负变正
        for (size_t i = 0; i < Lanes; ++i) {
            src[i] = (i % 2 == 0) ? TYPE_T(i + 1) : TYPE_T(-(static_cast<int>(i) + 1));
        }
        ns::store(t, dst.data(), ns::neg(t, ns::load(t, src.data())));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], static_cast<TYPE_T>(-src[i]));
        }

        // 2. 双重否定：-(-x) == x
        ns::store(t, dst.data(), ns::neg(t, ns::neg(t, ns::load(t, src.data()))));
        for (size_t i = 0; i < Lanes; ++i)
        {
            EXPECT_TRUE(dst[i] == src[i]);
        }

        // 3. 零值符号位测试
        ns::store(t, src.data(), ns::set(t, TYPE_T(0)));
        ns::store(t, dst.data(), ns::neg(t, ns::load(t, src.data())));
        for (size_t i = 0; i < Lanes; ++i) {
            EXPECT_EQ(dst[i], TYPE_T(0));

            #if KSIMD_TEST_FP
            {
                // IEEE 754: 0.0 -> -0.0
                EXPECT_NE(ksimd::sign_bit(src[i]), ksimd::sign_bit(dst[i]));
            }
            #endif
        }

        // 5. 浮点数特殊值
        #if KSIMD_TEST_FP
        {
            // Inf -> -Inf
            ns::store(t, dst.data(), ns::neg(t, ns::set(t, ksimd::Inf<TYPE_T>)));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_inf(dst[i]) && ksimd::sign_bit(dst[i]));
            }

            // NaN sign flip
            ns::Batch<decltype(t)> v_nan = ns::set(t, ksimd::QNaN<TYPE_T>);
            ns::store(t, src.data(), v_nan);
            ns::store(t, dst.data(), ns::neg(t, v_nan));
            for (size_t i = 0; i < Lanes; ++i) {
                EXPECT_TRUE(ksimd::is_NaN(dst[i]));
                EXPECT_NE(ksimd::sign_bit(src[i]), ksimd::sign_bit(dst[i]));
            }
        }
        #endif
    }
}
#if KSIMD_ONCE
TEST_ONCE_DYN(neg)
#endif

#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
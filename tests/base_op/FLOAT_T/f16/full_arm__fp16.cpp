#include <kSimd/core/impl/base.hpp>

#if KSIMD_ARCH_ARM_ANY

#include <cfloat>
#include <cstddef>
#include <limits>

#pragma message("test arm __fp16.")

#define TAG_T ns::FullTag<FLOAT_T>

using FLOAT_T = __fp16;
// constexpr size_t ALIGNMENT = 64;

#define FLOAT_T_EPSILON (__FLT16_EPSILON__)
#define FLOAT_T_EPSILON_RSQRT 1e-3f
#define FLOAT_T_EPSILON_RCP 1e-3f

#define FLOAT_T_EQ EXPECT_FLOAT_EQ

#define KSIMD_TEST_FP16 1

#if !KSIMD_SUPPORT_NATIVE_FP16
#error must support fp16
#endif
#include "../floating_point.inl"

#else
int main()
{
    return 0;
}
#endif
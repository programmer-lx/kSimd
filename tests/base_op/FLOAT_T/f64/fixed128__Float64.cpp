#include <kSimd/core/impl/base.hpp>

#if KSIMD_SUPPORT_EXTENSION_FLOAT64

#pragma message("test _Float64.")

#include <cfloat>
#include <cstddef>

#define TAG_T ns::Fixed128Tag<FLOAT_T>

using FLOAT_T = _Float64;
// constexpr size_t ALIGNMENT = 64;

#define FLOAT_T_EPSILON (FLT_EPSILON)
#define FLOAT_T_EPSILON_RSQRT 1e-3f
#define FLOAT_T_EPSILON_RCP 1e-3f

#define FLOAT_T_EQ EXPECT_FLOAT_EQ

// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_F64 1
#include "../floating_point.inl"

#else
int main()
{
    return 0;
}
#endif
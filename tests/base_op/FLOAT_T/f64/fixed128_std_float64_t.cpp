#if __STDCPP_FLOAT64_T__
#include <cfloat>
#include <cstddef>
#include <stdfloat>

#pragma message("test std::float64_t.")

#define TAG_T ns::Fixed128Tag<FLOAT_T>

using FLOAT_T = std::float64_t;
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
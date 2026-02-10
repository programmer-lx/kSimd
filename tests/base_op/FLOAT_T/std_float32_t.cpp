#if __STDCPP_FLOAT32_T__
#include <cfloat>
#include <cstddef>
#include <stdfloat>

using FLOAT_T = std::float32_t;
#define FLOAT_T_EPSILON (FLT_EPSILON)
#define FLOAT_T_EPSILON_RSQRT 1e-3f
#define FLOAT_T_EPSILON_RCP 1e-3f

#define FLOAT_T_EQ EXPECT_FLOAT_EQ

// constexpr size_t ALIGNMENT = 32;

#include "floating_point.inl"

#else
int main()
{
    return 0;
}
#endif
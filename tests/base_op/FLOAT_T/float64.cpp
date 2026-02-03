#include <cfloat>
#include <cstddef>

using FLOAT_T = double;
#define FLOAT_C(x) x
#define FLOAT_T_EPSILON (DBL_EPSILON)
#define FLOAT_T_EPSILON_RSQRT (DBL_EPSILON * 10.0)
#define FLOAT_T_EPSILON_ONE_DIV (DBL_EPSILON * 10.0)

#define FLOAT_T_EQ EXPECT_DOUBLE_EQ

constexpr size_t ALIGNMENT = 32;

#include "floating_point.inl"

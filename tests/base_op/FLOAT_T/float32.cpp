#include <cfloat>
#include <cstddef>

using FLOAT_T = float;
#define FLOAT_C(x) x##f
#define FLOAT_T_EPSILON (FLT_EPSILON)
#define FLOAT_T_EPSILON_RSQRT 1e-3f
#define FLOAT_T_EPSILON_ONE_DIV 1e-3f

#define FLOAT_T_EQ EXPECT_FLOAT_EQ

constexpr size_t ALIGNMENT = 32;

#include "floating_point.inl"

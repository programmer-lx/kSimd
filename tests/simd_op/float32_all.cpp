#include <cfloat>

using FLOAT_T = float;
#define FLOAT_T_EPSILON (FLT_EPSILON * 10.0f)
#define FLOAT_T_EPSILON_RSQRT 1e-3f
#define FLOAT_T_EPSILON_ONE_DIV 1e-3f

#include "floating_point_all.inl"

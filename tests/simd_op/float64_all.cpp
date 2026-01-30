#include <cfloat>

using FLOAT_T = double;
#define FLOAT_T_EPSILON (DBL_EPSILON * 10.0)
#define FLOAT_T_EPSILON_RSQRT (DBL_EPSILON * 10.0)
#define FLOAT_T_EPSILON_ONE_DIV (DBL_EPSILON * 10.0)

#include "floating_point_all.inl"

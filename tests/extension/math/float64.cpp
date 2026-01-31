#include <cfloat>
#include <cstddef>

using FLOAT_T = double;

#define FLOAT_C(x) x
#define FLOAT_T_EPSILON (DBL_EPSILON * 10.0)

constexpr size_t TOTAL = 16;
constexpr size_t ALIGNMENT = 32;

#include "floating_point.inl"
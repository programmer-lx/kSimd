#include <cfloat>
#include <cstddef>

using FLOAT_T = float;

#define FLOAT_C(x) x##f
#define FLOAT_T_EPSILON (FLT_EPSILON * 10.0f)

constexpr size_t ALIGNMENT = 32;

#include "floating_point.inl"

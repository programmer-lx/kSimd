#include <cfloat>
#include <cstddef>
#include <limits>

#define TAG_T ns::FullTag<FLOAT_T>

using FLOAT_T = float;
constexpr size_t ALIGNMENT = 64;

#include "vmath.inl"

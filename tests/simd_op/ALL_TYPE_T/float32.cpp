#include <cfloat>
#include <cstddef>

using TYPE_T = float;
#define VALUE_C(x) x##f
constexpr size_t TOTAL = 16;
constexpr size_t ALIGNMENT = 32;

#include "all_type.inl"

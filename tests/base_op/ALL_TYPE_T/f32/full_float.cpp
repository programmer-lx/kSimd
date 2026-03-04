#include <cfloat>
#include <cstddef>
#include <limits>

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = float;
constexpr size_t ALIGNMENT = 64;

#include "../all_type.inl"

#include <cfloat>
#include <cstddef>
#include <limits>
#include <cstdint>

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = uint64_t;
// constexpr size_t ALIGNMENT = 64;

#include "../all_type.inl"

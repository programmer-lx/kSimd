#include <cfloat>
#include <cstddef>
#include <limits>
#include <cstdint>

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = int64_t;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_SINT 1
#define KSIMD_TEST_SIGNED 1
#include "../all_type.inl"

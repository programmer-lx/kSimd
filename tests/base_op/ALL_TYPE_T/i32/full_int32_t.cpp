#include <cfloat>
#include <cstddef>
#include <limits>
#include <cstdint>

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = int32_t;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_SINT 1
#define KSIMD_TEST_SIGNED 1
#include "../all_type.inl"

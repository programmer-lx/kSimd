#include <cfloat>
#include <cstddef>
#include <limits>

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = float;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP32 1
#include "../signed.inl"

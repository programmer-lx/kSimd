#include <cfloat>
#include <cstddef>
#include <limits>

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = double;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP64 1
#include "../signed.inl"

#include <kSimd/core/impl/base.hpp>

#if KSIMD_SUPPORT_EXTENSION_FLOAT64

#include <cfloat>
#include <cstddef>
#include <limits>

#pragma message("test _Float64.")

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = _Float64;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP64 1
#include "../all_type.inl"

#else
int main()
{
    return 0;
}
#endif
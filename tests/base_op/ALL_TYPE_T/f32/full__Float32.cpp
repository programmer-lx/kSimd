#include <kSimd/core/impl/base.hpp>

#if KSIMD_SUPPORT_EXTENSION_FLOAT32

#include <cfloat>
#include <cstddef>
#include <limits>

#pragma message("test _Float32.")

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = _Float32;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP32 1
#include "../all_type.inl"

#else
int main()
{
    return 0;
}
#endif
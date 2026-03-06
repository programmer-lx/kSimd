#include <kSimd/core/impl/base.hpp>

#if KSIMD_SUPPORT_EXTENSION_FLOAT16

#include <cfloat>
#include <cstddef>
#include <limits>

#pragma message("test _Float16.")

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = _Float16;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP16 1
#include "../all_type.inl"

#else
int main()
{
    return 0;
}
#endif
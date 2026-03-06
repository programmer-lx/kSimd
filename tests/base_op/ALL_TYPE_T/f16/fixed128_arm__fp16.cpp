#include <kSimd/core/impl/base.hpp>

#if KSIMD_ARCH_ARM_ANY

#include <cfloat>
#include <cstddef>
#include <limits>

#pragma message("test arm __fp16.")

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = __fp16;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP16 1
#include "../all_type.inl"

#else
int main()
{
    return 0;
}
#endif
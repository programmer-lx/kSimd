#include <kSimd/core/impl/base.hpp>

#if KSIMD_SUPPORT_EXTENSION_FLOAT32

#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdfloat>

#pragma message("test _Float32.")

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = _Float32;
constexpr size_t ALIGNMENT = 64;

#include "../signed.inl"

#else
int main()
{
    return 0;
}
#endif
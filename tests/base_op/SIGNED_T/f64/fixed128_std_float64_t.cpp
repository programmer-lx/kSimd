#if __STDCPP_FLOAT64_T__

#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdfloat>

#pragma message("test std::float64_t.")

#define TAG_T ns::Fixed128Tag<TYPE_T>

using TYPE_T = std::float64_t;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP64 1
#include "../signed.inl"

#else

int main()
{
    return 0;
}

#endif
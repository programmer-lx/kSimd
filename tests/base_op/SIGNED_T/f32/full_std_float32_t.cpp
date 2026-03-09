#if __STDCPP_FLOAT32_T__

#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdfloat>

#pragma message("test std::float32_t.")

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = std::float32_t;
// constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP 1
#define KSIMD_TEST_FP32 1
#include "../signed.inl"

#else

int main()
{
    return 0;
}

#endif
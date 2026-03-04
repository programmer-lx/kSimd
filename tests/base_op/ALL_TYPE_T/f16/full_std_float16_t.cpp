#if __STDCPP_FLOAT16_T__
#include <cfloat>
#include <cstddef>
#include <limits>

#include <stdfloat>

#pragma message("test std::float16_t.")

#define TAG_T ns::FullTag<TYPE_T>

using TYPE_T = std::float16_t;
constexpr size_t ALIGNMENT = 64;

#define KSIMD_TEST_FP16 1
#include "../all_type.inl"

#else
int main()
{
    return 0;
}
#endif
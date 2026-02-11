#if __STDCPP_FLOAT32_T__
#include <cfloat>
#include <cstddef>
#include <limits>

#include <stdfloat>

#pragma message("test std::float32_t.")

using TYPE_T = std::float32_t;

#include "all_type.inl"

#else
int main()
{
    return 0;
}
#endif
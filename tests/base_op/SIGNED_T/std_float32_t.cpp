#if __STDCPP_FLOAT32_T__

#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdfloat>

using TYPE_T = std::float32_t;
// constexpr size_t ALIGNMENT = 32;

#include "signed.inl"

#else

int main()
{
    return 0;
}

#endif
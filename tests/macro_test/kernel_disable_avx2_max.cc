#include <array>
#include <stdexcept>
#include <string>

// #undef KSIMD_IS_TESTING
// #define KSIMD_IS_TESTING

// disable avx2_max before include <kSimd/core/impl/base.hpp>
#undef KSIMD_DISABLE_AVX2_MAX
#define KSIMD_DISABLE_AVX2_MAX

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_disable_avx2_max.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_disable_avx2_max()
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        if (str != KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR))
        {
            throw std::runtime_error("str only can be \"SCALAR\"");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_disable_avx2_max);
void kernel_disable_avx2_max()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_avx2_max));
    if (table_size != 1)
    {
        throw std::runtime_error("we should disable AVX2_MAX");
    }

    if (ksimd::detail::dyn_func_index() != 0)
    {
        throw std::runtime_error("index of scalar must be 0");
    }

    KSIMD_DYN_CALL(kernel_disable_avx2_max)();
}
#endif

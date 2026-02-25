#include <array>
#include <stdexcept>
#include <string>

// #undef KSIMD_IS_TESTING
// #define KSIMD_IS_TESTING

// disable avx2_max before include <kSimd/core/impl/base.hpp>
#undef KSIMD_DISABLE_SSE4_1
#define KSIMD_DISABLE_SSE4_1

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_disable_sse4_1.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

#if defined(KSIMD_IS_TESTING)
    #error no KSIMD_IS_TESTING
#endif

#pragma message("disable sse4.1")

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_disable_sse4_1(size_t index)
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_AVX2_FMA3) && index == 0);
        if (!result)
        {
            throw std::runtime_error("we should disable sse4.1");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_disable_sse4_1);
void kernel_disable_sse4_1()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_sse4_1));
    if (table_size != 2)
    {
        throw std::runtime_error("we should disable SSE4_1");
    }

    if (ksimd::detail::dyn_func_index() != 0)
    {
        throw std::runtime_error("index of sse4.1 must be 0");
    }

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_sse4_1)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_sse4_1)[i](i);
    }
}
#endif

#include <array>
#include <stdexcept>
#include <string>

// #undef KSIMD_IS_TESTING
// #define KSIMD_IS_TESTING

// disable avx2_max before include <kSimd/core/impl/base.hpp>
#undef KSIMD_DISABLE_X86_V2
#define KSIMD_DISABLE_X86_V2

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_disable_x86_v2.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

#if defined(KSIMD_IS_TESTING)
    #error no KSIMD_IS_TESTING
#endif

#pragma message("disable x86 v2")

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_disable_x86_v2(size_t index)
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V3) && index == 0);
        if (!result)
        {
            throw std::runtime_error("we should disable x86 v2");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_disable_x86_v2);
void kernel_disable_x86_v2()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v2));
    if (table_size != 2)
    {
        throw std::runtime_error("we should disable x86 v2");
    }

    if (ksimd::detail::dyn_func_index() != 0)
    {
        throw std::runtime_error("index of x86 v2 must be 0");
    }

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v2)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v2)[i](i);
    }
}
#endif

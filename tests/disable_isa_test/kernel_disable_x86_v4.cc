#include <array>
#include <stdexcept>
#include <string>


// disable avx512 v4 before include <kSimd/core/impl/base.hpp>
#undef KSIMD_DISABLE_X86_V4
#define KSIMD_DISABLE_X86_V4

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_disable_x86_v4.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

#if defined(KSIMD_IS_TESTING)
    #error no KSIMD_IS_TESTING
#endif

#pragma message("disable x86 v4")

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_disable_x86_v4(size_t index)
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 2) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V2) && index == 1) ||
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_X86_V3) && index == 0);
        if (!result)
        {
            throw std::runtime_error("we should disable x86 v4");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_disable_x86_v4);
void kernel_disable_x86_v4()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v4));
    if (table_size != 3)
    {
        throw std::runtime_error("we should disable x86 v4");
    }

    if (ksimd::detail::dyn_func_index() != 0)
    {
        throw std::runtime_error("index of x86 v4 must be 0");
    }

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v4)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_x86_v4)[i](i);
    }
}
#endif

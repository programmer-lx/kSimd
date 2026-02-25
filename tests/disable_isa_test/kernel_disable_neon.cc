#include <array>
#include <stdexcept>
#include <string>


// disable before include <kSimd/core/impl/base.hpp>
#undef KSIMD_DISABLE_NEON
#define KSIMD_DISABLE_NEON

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_disable_neon.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

#if defined(KSIMD_IS_TESTING)
    #error no KSIMD_IS_TESTING
#endif

#pragma message("disable neon")

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_disable_neon(size_t index)
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        bool result =
            (str == KSIMD_STR(KSIMD_DYN_INSTRUCTION_SCALAR) && index == 0);
        if (!result)
        {
            throw std::runtime_error("we should disable neon.");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_disable_neon);
void kernel_disable_neon()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_neon));
    if (table_size != 1)
    {
        throw std::runtime_error("we should disable NEON");
    }

    if (ksimd::detail::dyn_func_index() != 0)
    {
        throw std::runtime_error("index of scalar must be 0");
    }

    // try call
    for (size_t i = 0; i < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_neon)); ++i)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_disable_neon)[i](i);
    }
}
#endif

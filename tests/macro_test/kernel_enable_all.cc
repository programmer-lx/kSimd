#include <array>
#include <stdexcept>

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "kernel_enable_all.cc"
#include <kSimd/core/dispatch_this_file.hpp>

#include "header.hpp"

namespace KSIMD_DYN_INSTRUCTION
{
    void kernel_enable_all()
    {
        std::string str = KSIMD_STR(KSIMD_DYN_INSTRUCTION);
        if (str != KSIMD_STR(KSIMD_DYN_INSTRUCTION_AVX2_MAX))
        {
            throw std::runtime_error("str only can be \"AVX2_MAX\"");
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(kernel_enable_all);
void kernel_enable_all()
{
    volatile size_t table_size = std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(kernel_enable_all));

    if (table_size != 2)
    {
        throw std::runtime_error("we should enable all intrinsics");
    }

    if (KSIMD_dyn_func_index() != 0)
    {
        throw std::runtime_error("index of AVX2_MAX must be 0");
    }

    KSIMD_DYN_CALL(kernel_enable_all)();
}
#endif

#include "header.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
    #if KSIMD_ARCH_X86_ANY
    try
    {
        kernel_disable_avx2_max();
        kernel_enable_all();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    #endif

    return EXIT_SUCCESS;
}
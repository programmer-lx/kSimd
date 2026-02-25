#include "header.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

#include <kSimd/macros.h>

#if defined(KSIMD_IS_TESTING)
    #error no KSIMD_IS_TESTING
#endif

int main()
{
    try
    {
        #if KSIMD_ARCH_X86_ANY

        kernel_disable_avx2_fma3();
        kernel_disable_sse4_1();

        #elif KSIMD_ARCH_ARM_64

        kernel_disable_neon();

        #else
        #error unknown arch
        #endif
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
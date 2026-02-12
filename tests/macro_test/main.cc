#include "header.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
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

    return EXIT_SUCCESS;
}
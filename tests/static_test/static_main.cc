#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#include "static_hpp_1.hpp"
#include "static_hpp_2.hpp"

int main()
{
    try
    {
        // static var test
        {
            StaticStruct* static_var_1 = get_static_var_1();
            StaticStruct* static_var_2 = get_static_var_2();
            volatile uintptr_t ptr1 = reinterpret_cast<uintptr_t>(static_var_1);
            volatile uintptr_t ptr2 = reinterpret_cast<uintptr_t>(static_var_2);
            if (ptr1 == ptr2) // static 是每个TU一份，所以指针不能相等
            {
                throw std::exception();
            }
        }

        // static function static var test
        {
            const StaticStruct* var1 = get_static_func_var_1();
            const StaticStruct* var2 = get_static_func_var_2();

            volatile uintptr_t ptr1 = reinterpret_cast<uintptr_t>(var1);
            volatile uintptr_t ptr2 = reinterpret_cast<uintptr_t>(var2);
            if (ptr1 == ptr2)
            {
                throw std::exception();
            }
        }
    }
    catch (...)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
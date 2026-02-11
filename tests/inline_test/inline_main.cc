#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#include "inline_hpp_1.hpp"
#include "inline_hpp_2.hpp"

int main()
{
    try
    {
        // inline var test
        {
            InlineStruct* inline_var_1 = get_inline_var_1();
            InlineStruct* inline_var_2 = get_inline_var_2();
            volatile uintptr_t ptr1 = reinterpret_cast<uintptr_t>(inline_var_1);
            volatile uintptr_t ptr2 = reinterpret_cast<uintptr_t>(inline_var_2);
            if (ptr1 != ptr2) // inline: 链接器会合并多个inline声明，所以地址必须相等
            {
                throw std::exception();
            }
        }

        // inline function static var test
        {
            const InlineStruct* var1 = get_inline_func_var_1();
            const InlineStruct* var2 = get_inline_func_var_2();

            volatile uintptr_t ptr1 = reinterpret_cast<uintptr_t>(var1);
            volatile uintptr_t ptr2 = reinterpret_cast<uintptr_t>(var2);
            if (ptr1 != ptr2) // inline: 链接器会合并多个inline声明，所以地址必须相等
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
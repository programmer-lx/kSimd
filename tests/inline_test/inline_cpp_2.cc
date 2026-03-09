#include "inline_hpp_2.hpp"

#include "inline_header.hpp"

InlineStruct* get_inline_var_2()
{
    return &inline_var;
}

const InlineStruct* get_inline_func_var_2()
{
    return &inline_static_struct();
}
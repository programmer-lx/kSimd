#include "static_hpp_2.hpp"

#include "static_header.hpp"

StaticStruct* get_static_var_2()
{
    return &static_var;
}

const StaticStruct* get_static_func_var_2()
{
    return &static_static_struct();
}
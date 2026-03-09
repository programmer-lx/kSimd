#pragma once

struct InlineStruct
{
    int a;
    float b;
    double c;
};

inline InlineStruct inline_var = {};

inline const InlineStruct& inline_static_struct() noexcept
{
    static InlineStruct s = {};
    return s;
}

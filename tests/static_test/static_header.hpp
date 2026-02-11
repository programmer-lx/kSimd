#pragma once

struct StaticStruct
{
    int a;
    float b;
    double c;
};

static StaticStruct static_var = {};

static const StaticStruct& static_static_struct() noexcept
{
    static StaticStruct s = {};
    return s;
}

#pragma once

namespace ksimd
{
    struct OpHelper
    {
        enum class RoundingMode
        {
            Up,         // 向上取整
            Down,       // 向下取整
            Nearest,    // 向最近偶数取整
            ToZero,     // 向0取整
            Round       // 四舍五入
        };
    };
}
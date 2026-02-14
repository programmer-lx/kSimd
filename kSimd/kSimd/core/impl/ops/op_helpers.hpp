#pragma once

namespace ksimd
{
    enum class RoundingMode
    {
        Up,         // 向上取整
        Down,       // 向下取整
        Nearest,    // 向最近偶数取整
        ToZero,     // 向0取整
        Round       // 四舍五入
    };

    enum class FloatMinMaxOption
    {
        Native,     // 如果右操作数是NaN，则返回NaN，否则返回左操作数
        CheckNaN    // 检查NaN的传播 (如果传入的值有一个NaN，则会返回NaN)
    };
}
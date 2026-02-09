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

    template<typename ScalarType, typename BatchType, typename MaskType, size_t VectorBytes, size_t Alignment_>
    struct OpInfo
    {
        using scalar_t = ScalarType;
        using batch_t = BatchType;
        using mask_t = MaskType;
        static constexpr size_t Lanes = VectorBytes / sizeof(scalar_t);
        static constexpr size_t Alignment = Alignment_;

        static_assert(VectorBytes % sizeof(scalar_t) == 0);
    };
}
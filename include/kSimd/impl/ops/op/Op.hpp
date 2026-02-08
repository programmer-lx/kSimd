#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction, is_scalar_type ScalarType>
struct Op;

#define KSIMD_DYN_OP(scalar_type) \
    KSIMD_NAMESPACE_NAME::Op<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, scalar_type>

struct OpHelper
{
    enum class RoundingMode : int
    {
        Nearest,    // 最近偶数
        Up,         // 向上取整
        Down,       // 向下取整
        ToZero,     // 向0取整
        Round       // 四舍五入
    };
};

KSIMD_NAMESPACE_END

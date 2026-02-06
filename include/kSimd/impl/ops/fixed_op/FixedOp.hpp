#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction, is_scalar_type ScalarType, size_t Width, size_t Count>
struct FixedOp;

#define KSIMD_DYN_FIXED_OP(scalar_type, width, count) \
    KSIMD_NAMESPACE_NAME::FixedOp<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, scalar_type, width, count>

// helper类，提供各种辅助操作(掩码等)，由最顶层FixedOp继承他
template<size_t Width>
struct FixedOpHelper;

template<>
struct FixedOpHelper<4>
{
    // masks
    static constexpr int All  = 0b1111;
    static constexpr int None = 0b0000;

    static constexpr int X    = 0b0001;
    static constexpr int Y    = 0b0010;
    static constexpr int Z    = 0b0100;
    static constexpr int W    = 0b1000;
};

KSIMD_NAMESPACE_END
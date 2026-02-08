#pragma once

#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    template<SimdInstruction I, is_scalar_type S, size_t Width>
    consteval size_t max_count_of_fixed_op()
    {
        constexpr size_t reg_lanes = RegLanes<I, S>;
        static_assert(reg_lanes % Width == 0);
        return reg_lanes / Width;
    }
}

template<SimdInstruction Instruction, is_scalar_type ScalarType, size_t Width, size_t Count>
struct PackedOp;

#define KSIMD_DETAIL_DYN_PACKED_OP_COUNT(scalar_type, width, count) \
    KSIMD_NAMESPACE_NAME::PackedOp<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, \
        scalar_type, width, count>


#define KSIMD_DETAIL_MAX_COUNT_OF_PACKED_OP(scalar_type, width) \
    KSIMD_NAMESPACE_NAME::detail::max_count_of_fixed_op<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION, \
        scalar_type, width>()

#define KSIMD_DYN_PACKED_OP(scalar_type, width) \
    KSIMD_DETAIL_DYN_PACKED_OP_COUNT(scalar_type, width, KSIMD_DETAIL_MAX_COUNT_OF_PACKED_OP(scalar_type, width))

template<size_t width, size_t count>
struct PackedOpInfo
{
    static constexpr size_t Width = width;
    static constexpr size_t Count = count;
};

// helper类，提供各种辅助操作(掩码等)，由最顶层FixedOp继承他
template<size_t Width>
struct PackedOpHelper;

template<>
struct PackedOpHelper<4>
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
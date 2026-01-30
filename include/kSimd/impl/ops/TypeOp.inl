#pragma once

#include "SimdTraits.inl"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction, typename = void>
struct TypeOp;

#define KSIMD_DYN_TYPE_OP() \
    KSIMD_NAMESPACE_NAME::TypeOp<KSIMD_NAMESPACE_NAME::SimdInstruction::KSIMD_DYN_INSTRUCTION>

#define KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(T_to, T_from, alignment) \
    /* 位宽一致 */ \
    static_assert( sizeof(T_to) == sizeof(T_from ), \
        "sizeof(To) == sizeof(From)."); \
    static_assert( sizeof(decltype(std::declval<T_to>().v)) == sizeof(decltype(std::declval<T_from>().v)), \
        "sizeof(To.v) == sizeof(From.v)."); \
    \
    /* 判断双方的对齐 */ \
    static_assert( (alignof(T_to) == (alignment)) && (alignof(T_from) == (alignment)), \
        "alignof(To) == alignof(From) == " #alignment "."); \
    static_assert( (alignof(T_to) - offsetof(T_to, v) == (alignment)) && (alignof(T_from) - offsetof(T_from, v) == (alignment)), \
        "alignof(To.v) == alignof(From.v) == " #alignment ".");

KSIMD_NAMESPACE_END

#pragma once

#include <bit> // std::bit_cast
#include <utility> // index_sequence

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct TypeOp<SimdInstruction::Scalar>
{
    // scalar array <- scalar array or self <- self
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray && From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SIG_SCALAR_STATIC(To, bit_cast, (From from))
    {
        static_assert(sizeof(To) == sizeof(From ), "sizeof(To) == sizeof(From).");
        static_assert(sizeof(decltype(std::declval<To>().v)) == sizeof(decltype(std::declval<From>().v)), "sizeof(To.v) == sizeof(From.v).");

        return std::bit_cast<To>(from);
    }
};

KSIMD_NAMESPACE_END

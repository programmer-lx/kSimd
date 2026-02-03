#pragma once

#include <bit> // std::bit_cast

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/ops/type_op/TypeOp.hpp"
#include "kSimd/impl/func_attr.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct TypeOp<SimdInstruction::Scalar>
{
    // scalar array <- scalar array or self <- self
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SCALAR_API static To KSIMD_CALL_CONV bit_cast(From from) noexcept
    {
        static_assert(sizeof(To) == sizeof(From), "sizeof(To) == sizeof(From).");
        static_assert(sizeof(decltype(std::declval<To>().v)) == sizeof(decltype(std::declval<From>().v)),
                      "sizeof(To.v) == sizeof(From.v).");

        return std::bit_cast<To>(from);
    }
};

KSIMD_NAMESPACE_END

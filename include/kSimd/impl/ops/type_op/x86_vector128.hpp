#pragma once

#include <bit> // std::bit_cast

#include "kSimd/impl/traits.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/type_op/TypeOp.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct TypeOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE>
{
    // scalar array <- scalar array (用于转换除了float32之外的类型) (a to b or self to self)
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SCALAR_API static To KSIMD_CALL_CONV bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return std::bit_cast<To>(from);
    }

#define KSIMD_API(ret) KSIMD_OP_SSE_API static ret KSIMD_CALL_CONV

    // m128 <- m128 (self to self)
    template<is_batch_type To, is_batch_type From>
        requires(std::is_same_v<To, From> && From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return from;
    }

    // scalar array <- m128
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        To result;
        _mm_storeu_ps(reinterpret_cast<float32*>(result.v[0]), from.v[0]);
        return result;
    }

    // m128 <- scalar array
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_loadu_ps(reinterpret_cast<const float32*>(from.v[0])) };
    }

#undef KSIMD_API
};

template<SimdInstruction I>
    requires(I >= SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2 && I < SimdInstruction::SSE_End)
struct TypeOp<I>
{
#define KSIMD_API(ret) KSIMD_OP_SSE2_API static ret KSIMD_CALL_CONV

    // self <- self
    template<is_batch_type To, is_batch_type From>
        requires(std::is_same_v<To, From> && From::underlying_simd_type != detail::UnderlyingSimdType::ScalarArray)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return from;
    }

    // m128d <- m128
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128d &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castps_pd(from.v[0]) };
    }

    // m128i <- m128
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castps_si128(from.v[0]) };
    }

    // m128 <- m128d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castpd_ps(from.v[0]) };
    }

    // m128i <- m128d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castpd_si128(from.v[0]) };
    }

    // m128 <- m128i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castsi128_ps(from.v[0]) };
    }

    // m128d <- m128i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m128d &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m128i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec128)

        return { _mm_castsi128_pd(from.v[0]) };
    }

#undef KSIMD_API
};

KSIMD_NAMESPACE_END

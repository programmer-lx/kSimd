#pragma once

#include <bit> // std::bit_cast

#include "_SSE_family_types.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct TypeOp<SimdInstruction::SSE>
{
    // scalar array <- scalar array (用于转换除了float32之外的类型) (a to b or self to self)
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray && From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SIG_SCALAR(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return std::bit_cast<To>(from);
    }

    // m128 <- m128 (self to self)
    template<is_simd_type To, is_simd_type From>
        requires (std::is_same_v<To, From> && From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_OP_SIG_SSE(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return from;
    }

    // scalar array <- m128
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray && From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_OP_SIG_SSE(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        To result;
        _mm_store_ps(reinterpret_cast<float*>(result.v), from.v);
        return result;
    }

    // m128 <- scalar array
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128 && From::underlying_simd_type == detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SIG_SSE(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_load_ps(reinterpret_cast<const float*>(from.v)) };
    }
};

template<SimdInstruction Instruction>
    requires (Instruction >= SimdInstruction::SSE2 && Instruction < SimdInstruction::SSE_End)
struct TypeOp<Instruction>
{
    // self <- self
    template<is_simd_type To, is_simd_type From>
        requires (std::is_same_v<To, From> && From::underlying_simd_type != detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return from;
    }

    // m128d <- m128
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128d && From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castps_pd(from.v) };
    }

    // m128i <- m128
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128i && From::underlying_simd_type == detail::UnderlyingSimdType::m128)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castps_si128(from.v) };
    }

    // m128 <- m128d
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128 && From::underlying_simd_type == detail::UnderlyingSimdType::m128d)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castpd_ps(from.v) };
    }

    // m128i <- m128d
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128i && From::underlying_simd_type == detail::UnderlyingSimdType::m128d)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castpd_si128(from.v) };
    }

    // m128 <- m128i
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128 && From::underlying_simd_type == detail::UnderlyingSimdType::m128i)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castsi128_ps(from.v) };
    }

    // m128d <- m128i
    template<is_simd_type To, is_simd_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m128d && From::underlying_simd_type == detail::UnderlyingSimdType::m128i)
    KSIMD_OP_SIG_SSE2(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, Alignment::SSE_Family)

        return { _mm_castsi128_pd(from.v) };
    }
};

KSIMD_NAMESPACE_END

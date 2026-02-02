#pragma once

#include "types.hpp"
#include "kSimd/impl/ops/TypeOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction Instruction>
    requires (Instruction > SimdInstruction::AVX_Start && Instruction < SimdInstruction::AVX_End)
struct TypeOp<Instruction>
{
    // self <- self
    template<is_batch_type To, is_batch_type From>
        requires (std::is_same_v<To, From> && From::underlying_simd_type != detail::UnderlyingSimdType::ScalarArray)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return from;
    }

    // m256d <- m256
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256d && From::underlying_simd_type == detail::UnderlyingSimdType::m256)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castps_pd(from.v) };
    }

    // m256i <- m256
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256i && From::underlying_simd_type == detail::UnderlyingSimdType::m256)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castps_si256(from.v) };
    }

    // m256 <- m256d
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256 && From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castpd_ps(from.v) };
    }

    // m256i <- m256d
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256i && From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castpd_si256(from.v) };
    }

    // m256 <- m256i
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256 && From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castsi256_ps(from.v) };
    }

    // m256d <- m256i
    template<is_batch_type To, is_batch_type From>
        requires (To::underlying_simd_type == detail::UnderlyingSimdType::m256d && From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_OP_SIG_AVX_STATIC(To, bit_cast, (From from))
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::AVX_Family)

        return { _mm256_castsi256_pd(from.v) };
    }
};

KSIMD_NAMESPACE_END

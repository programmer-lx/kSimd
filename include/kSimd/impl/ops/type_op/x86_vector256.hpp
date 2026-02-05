#pragma once

#include "kSimd/impl/ops/type_op/TypeOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/traits.hpp"

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I>
    requires(I > SimdInstruction::AVX_Start && I < SimdInstruction::AVX_End)
struct TypeOp<I>
{
#define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV

    // self <- self
    template<is_batch_type To, is_batch_type From>
        requires(std::is_same_v<To, From> && From::underlying_simd_type != detail::UnderlyingSimdType::ScalarArray)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return from;
    }

    // m256d <- m256
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256d &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castps_pd(from.v[0]) };
    }

    // m256i <- m256
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castps_si256(from.v[0]) };
    }

    // m256 <- m256d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castpd_ps(from.v[0]) };
    }

    // m256i <- m256d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castpd_si256(from.v[0]) };
    }

    // m256 <- m256i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castsi256_ps(from.v[0]) };
    }

    // m256d <- m256i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256d &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        return { _mm256_castsi256_pd(from.v[0]) };
    }

#undef KSIMD_API
};

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/ops/type_op/TypeOp.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/traits.hpp"

#define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_F16C_API static ret KSIMD_CALL_CONV

KSIMD_NAMESPACE_BEGIN

template<SimdInstruction I>
    requires(I > SimdInstruction::AVX_Start && I < SimdInstruction::AVX_End)
struct TypeOp<I>
{
    // bit_cast (内存重解释)
    #define KSIMD_BIT_CAST_IMPL(intrinsic) \
        static_assert(To::reg_count <= 2, "512 bit is max."); \
        if constexpr (To::reg_count == 1) { return { intrinsic(from.v[0]) }; } \
        else { return { intrinsic(from.v[0]), intrinsic(from.v[1]) }; }

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

        KSIMD_BIT_CAST_IMPL(_mm256_castps_pd)
    }

    // m256i <- m256
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        KSIMD_BIT_CAST_IMPL(_mm256_castps_si256)
    }

    // m256 <- m256d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        KSIMD_BIT_CAST_IMPL(_mm256_castpd_ps)
    }

    // m256i <- m256d
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256i &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256d)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        KSIMD_BIT_CAST_IMPL(_mm256_castpd_si256)
    }

    // m256 <- m256i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256 &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        KSIMD_BIT_CAST_IMPL(_mm256_castsi256_ps)
    }

    // m256d <- m256i
    template<is_batch_type To, is_batch_type From>
        requires(To::underlying_simd_type == detail::UnderlyingSimdType::m256d &&
                 From::underlying_simd_type == detail::UnderlyingSimdType::m256i)
    KSIMD_API(To) bit_cast(From from) noexcept
    {
        KSIMD_DETAIL_TYPE_OP_BITCAST_CHECK(To, From, alignment::Vec256)

        KSIMD_BIT_CAST_IMPL(_mm256_castsi256_pd)
    }

    #undef KSIMD_BIT_CAST_IMPL

    // convert (等宽类型转换，比如 int32 -> float32)

    // promote_(low, high) (转换为高位宽类型，只能转换前半部分或后半部分，比如 float16 -> float32)

    // demote (转换为低位宽类型，转换后的类型，只有前半部分有效，后半部分用0填充，比如 float32 -> float16)

};

KSIMD_NAMESPACE_END

#undef KSIMD_API

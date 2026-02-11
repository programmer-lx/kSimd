#pragma once

#include <cstddef>
#include <cstdint>

#include <limits>
#include <type_traits>

#include "base.hpp"

#if KSIMD_SUPPORT_STD_FLOAT16 || KSIMD_SUPPORT_STD_BFLOAT16 || KSIMD_SUPPORT_STD_FLOAT32 || KSIMD_SUPPORT_STD_FLOAT64
    #include <stdfloat>
#endif

// clang-format off

namespace ksimd
{
    // ----------------- scalar type -----------------

    // floating point
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

    template<typename T>
    concept is_scalar_floating_point =
           std::is_same_v<T, float>
        || std::is_same_v<T, double>
#if KSIMD_SUPPORT_STD_FLOAT32
        || std::is_same_v<T, std::float32_t>
#endif
#if KSIMD_SUPPORT_STD_FLOAT64
        || std::is_same_v<T, std::float64_t>
#endif
        ;

    template<typename T>
    concept is_scalar_type =
        is_scalar_floating_point<T>   ||
        std::is_same_v<T, int8_t>     ||
        std::is_same_v<T, uint8_t>    ||
        std::is_same_v<T, int16_t>    ||
        std::is_same_v<T, uint16_t>   ||
        std::is_same_v<T, int32_t>    ||
        std::is_same_v<T, uint32_t>   ||
        std::is_same_v<T, int64_t>    ||
        std::is_same_v<T, uint64_t>;

    template<typename T, typename... Ts>
    concept is_scalar_type_includes = is_scalar_type<T> && (std::is_same_v<T, Ts> || ...);

    // signed types
    template<typename T>
    concept is_scalar_signed = is_scalar_type<T> && std::is_signed_v<T>;

    template<typename T, typename... Ts>
    concept is_scalar_signed_includes = is_scalar_signed<T> && (std::is_same_v<T, Ts> || ...);

    // float32
    template<typename T>
    concept is_scalar_type_float_32bits = is_scalar_type_includes<
        T
        , float
    #if KSIMD_SUPPORT_STD_FLOAT32
        , std::float32_t
    #endif
    >;

    // float64
    template<typename T>
    concept is_scalar_type_float_64bits = is_scalar_type_includes<
        T
        , double
    #if KSIMD_SUPPORT_STD_FLOAT64
        , std::float64_t
    #endif
    >;

    namespace alignment
    {
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec128 = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec256 = 32;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec512 = 64;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Max    = Vec512;
    }

    namespace vec_size
    {
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec128 = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec256 = 32;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec512 = 64;
    }
}

// clang-format on

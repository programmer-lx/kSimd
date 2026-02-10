#pragma once

#include <cstdint>

#include <limits>
#include <type_traits>

#include "platform.hpp"

namespace ksimd
{
    // clang-format off

    // ----------------- scalar type -----------------

    // (u)int(n)
    using int8   = int8_t   ;
    using uint8  = uint8_t  ;
    using int16  = int16_t  ;
    using uint16 = uint16_t ;
    using int32  = int32_t  ;
    using uint32 = uint32_t ;
    using int64  = int64_t  ;
    using uint64 = uint64_t ;

    // floating point
    enum class float16 : uint16 {};
    using float32 = float;
    using float64 = double;
    static_assert(sizeof(float32) == 4 && std::numeric_limits<float32>::is_iec559);
    static_assert(sizeof(float64) == 8 && std::numeric_limits<float64>::is_iec559);

    // clang-format on

    template<typename T>
    concept is_scalar_floating_point = std::is_same_v<T, float16> || std::is_same_v<T, float32> || std::is_same_v<T, float64>;

    template<typename T>
    concept is_scalar_type =
        is_scalar_floating_point<T> ||
        std::is_same_v<T, int8>     ||
        std::is_same_v<T, uint8>    ||
        std::is_same_v<T, int16>    ||
        std::is_same_v<T, uint16>   ||
        std::is_same_v<T, int32>    ||
        std::is_same_v<T, uint32>   ||
        std::is_same_v<T, int64>    ||
        std::is_same_v<T, uint64>;

    template<typename T, typename... Ts>
    concept is_scalar_type_includes = is_scalar_type<T> && (std::is_same_v<T, Ts> || ...);

    template<typename T>
    concept is_scalar_signed = is_scalar_type<T> && std::is_signed_v<T>;

    template<typename T, typename... Ts>
    concept is_scalar_signed_includes = is_scalar_signed<T> && (std::is_same_v<T, Ts> || ...);
}

#pragma once

#include <type_traits>

#include "common_macros.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    template<typename T>
    using underlying_t =
        std::conditional_t<
            std::is_enum_v<T>,
            std::underlying_type_t<T>,
            T
        >;

    template<typename T>
        requires (std::is_enum_v<T> || std::is_integral_v<T>)
    constexpr underlying_t<T> underlying(const T val) noexcept
    {
        return static_cast<underlying_t<T>>(val);
    }
}

KSIMD_NAMESPACE_END

#pragma once

#include <bit>
#include <type_traits>
#include <limits>

#include "common_macros.hpp"
#include "traits.hpp"

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

    template<size_t Bytes>
    struct uint_from_bytes
    {
        using type = std::conditional_t<
            (Bytes == sizeof(uint8)), uint8,
            std::conditional_t<
                (Bytes == sizeof(uint16)), uint16,
                std::conditional_t<
                    (Bytes == sizeof(uint32)), uint32,
                    std::conditional_t<
                        (Bytes == sizeof(uint64)), uint64, void
                    >
                >
            >
        >;

        // check
        static_assert(!std::is_void_v<type>);
    };

    template<size_t Bytes>
    using uint_from_bytes_t = typename uint_from_bytes<Bytes>::type;

    template<typename S, int index>
    consteval int inverse_bit_index_impl()
    {
        constexpr int idx = static_cast<int>(sizeof(S)) * 8 - index - 1;
        static_assert(idx >= 0 && idx < sizeof(S) * 8);
        return idx;
    }

    template<is_scalar_floating_point F>
    consteval F exp_mask()
    {
        if constexpr (std::is_same_v<float32, F>)
        {
            constexpr uint32 mask = static_cast<uint32>(0b1111'1111u) << 23;
            return std::bit_cast<F>(mask);
        }
        else if constexpr (std::is_same_v<float64, F>)
        {
            constexpr uint64 mask = static_cast<uint64>(0b111'1111'1111u) << 52;
            return std::bit_cast<F>(mask);
        }
        else
        {
            return 0;
        }
    }
} // namespace detail

enum class RoundingMode : int
{
    Nearest,    // 最近偶数
    Up,         // 向上取整
    Down,       // 向下取整
    ToZero,     // 向0取整
    Round       // 四舍五入
};

template<is_scalar_type S>
using same_bits_uint_t = detail::uint_from_bytes_t<sizeof(S)>;

template<is_scalar_type S>
KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr auto bitcast_to_uint(S n) noexcept
{
    return std::bit_cast<same_bits_uint_t<S>>(n);
}

template<is_scalar_type S>
KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr S min(S a, S b) noexcept
{
    return a < b ? a : b;
}

template<is_scalar_type S>
KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr S max(S a, S b) noexcept
{
    return a > b ? a : b;
}

KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_NaN(const float32 f) noexcept
{
    // 指数位均为1，尾数位 != 0
    // 1位符号，8位指数，23位尾数
    const uint32 bits = std::bit_cast<uint32>(f);
    constexpr uint32 exp_mask = static_cast<uint32>(0b1111'1111u) << 23;
    constexpr uint32 mantissa_mask = ~static_cast<uint32>(0u) >> 9;
    return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0u);
}

KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_NaN(const float64 f) noexcept
{
    // 1位符号位，11位指数，52位尾数
    const uint64 bits = std::bit_cast<uint64>(f);
    constexpr uint64 exp_mask = static_cast<uint64>(0b111'1111'1111u) << 52;
    constexpr uint64 mantissa_mask = ~static_cast<uint64>(0u) >> 12;
    return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0ull);
}

KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_finite(const float32 f) noexcept
{
    // 指数位不全为1
    const uint32 bits = std::bit_cast<uint32>(f);
    constexpr uint32 exp_mask = static_cast<uint32>(0b1111'1111u) << 23;
    return (bits & exp_mask) != exp_mask;
}

KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_finite(const float64 f) noexcept
{
    const uint64 bits = std::bit_cast<uint64>(f);
    constexpr uint64 exp_mask = static_cast<uint64>(0b111'1111'1111u) << 52;
    return (bits & exp_mask) != exp_mask;
}

template<typename S, int index>
KSIMD_HEADER_GLOBAL_CONSTEXPR int InverseBitIndex = detail::inverse_bit_index_impl<S, index>();

/**
 * @brief 0b11111111...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S OneBlock = std::bit_cast<S>(~static_cast<detail::uint_from_bytes_t<sizeof(S)>>(0));

/**
 * @brief 0b00000000...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S ZeroBlock = static_cast<S>(0);

/**
 * @brief 0b10000000...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S SignBitMask = std::bit_cast<S>(
    static_cast<detail::uint_from_bytes_t<sizeof(S)>>(1) << static_cast<detail::uint_from_bytes_t<sizeof(S)>>(InverseBitIndex<S, 0>)
);

/**
 * @brief 0b01111111...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S SignBitClearMask = std::bit_cast<S>(OneBlock<same_bits_uint_t<S>> >> static_cast<same_bits_uint_t<S>>(1));

/**
 * @brief 指数位全为1，其余位全为0
 */
template<is_scalar_floating_point F>
KSIMD_HEADER_GLOBAL_CONSTEXPR F ExpMask = detail::exp_mask<F>();

template<is_scalar_floating_point F>
KSIMD_HEADER_GLOBAL_CONSTEXPR F Inf = std::numeric_limits<F>::infinity();

template<is_scalar_floating_point F>
KSIMD_HEADER_GLOBAL_CONSTEXPR F QNaN = std::numeric_limits<F>::quiet_NaN();

KSIMD_NAMESPACE_END

#pragma once

#include <bit>

#include "../common_macros.hpp"
#include "SimdTraits.inl"
#include "utils.inl"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
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

    template<is_scalar_type S>
    using same_bits_uint_t = uint_from_bytes_t<sizeof(S)>;

    template<is_scalar_type S>
    constexpr KSIMD_FORCE_INLINE auto bitcast_to_uint(S n) noexcept
    {
        return std::bit_cast<same_bits_uint_t<S>>(n);
    }

    template<is_scalar_type S>
    constexpr KSIMD_FORCE_INLINE S min(S a, S b) noexcept
    {
        return a < b ? a : b;
    }

    template<is_scalar_type S>
    constexpr KSIMD_FORCE_INLINE S max(S a, S b) noexcept
    {
        return a > b ? a : b;
    }

    template<is_scalar_type S>
    constexpr KSIMD_FORCE_INLINE S unsafe_clamp(S v, S min, S max) noexcept
    {
        return detail::min(detail::max(v, min), max);
    }

    template<is_scalar_type S>
    constexpr KSIMD_FORCE_INLINE S safe_clamp(S v, S range1, S range2) noexcept
    {
        S low = detail::min(range1, range2);
        S high = detail::max(range1, range2);
        return detail::min(detail::max(v, low), high);
    }

    template<typename S, int index>
    consteval int inverse_bit_index_impl()
    {
        constexpr int idx = static_cast<int>(sizeof(S)) * 8 - index - 1;
        static_assert(idx >= 0 && idx < sizeof(S) * 8);
        return idx;
    }
} // namespace detail

template<typename S, int index>
KSIMD_HEADER_GLOBAL_CONSTEXPR int inverse_bit_index = detail::inverse_bit_index_impl<S, index>();

/**
 * @return 0b11111111...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S one_block = std::bit_cast<S>(~static_cast<detail::uint_from_bytes_t<sizeof(S)>>(0));

/**
 * @return 0b00000000...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S zero_block = static_cast<S>(0);

/**
 * @return 0b10000000...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S sign_bit_mask = std::bit_cast<S>(
    static_cast<detail::uint_from_bytes_t<sizeof(S)>>(1) << static_cast<detail::uint_from_bytes_t<sizeof(S)>>(inverse_bit_index<S, 0>)
);

/**
 * @return 0b01111111...
 */
template<is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR S sign_bit_clear_mask = std::bit_cast<S>(one_block<detail::same_bits_uint_t<S>> >> static_cast<detail::same_bits_uint_t<S>>(1));

KSIMD_NAMESPACE_END

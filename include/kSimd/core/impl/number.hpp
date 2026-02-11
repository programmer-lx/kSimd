#pragma once

#include <cstdint>

#include <bit>
#include <type_traits>
#include <limits>

#include "base.hpp"
#include "types.hpp"

namespace ksimd
{
    namespace detail
    {
        template<size_t Bytes>
        struct uint_from_bytes;

        template<>
        struct uint_from_bytes<1>
        {
            using type = uint8_t;
        };

        template<>
        struct uint_from_bytes<2>
        {
            using type = uint16_t;
        };

        template<>
        struct uint_from_bytes<4>
        {
            using type = uint32_t;
        };

        template<>
        struct uint_from_bytes<8>
        {
            using type = uint64_t;
        };

        template<typename S, int index>
        consteval int inverse_bit_index_impl()
        {
            constexpr int idx = static_cast<int>(sizeof(S)) * 8 - index - 1;
            static_assert(idx >= 0 && idx < sizeof(S) * 8);
            return idx;
        }

        template<is_scalar_type_float_32bits F>
        consteval F exp_mask()
        {
            // 1位符号，8位指数，23位尾数
            constexpr uint32_t mask = UINT32_C(0b1111'1111) << 23;
            return std::bit_cast<F>(mask);
        }

        template<is_scalar_type_float_64bits F>
        consteval F exp_mask()
        {
            // 1位符号位，11位指数，52位尾数
            constexpr uint64_t mask = UINT64_C(0b111'1111'1111) << 52;
            return std::bit_cast<F>(mask);
        }

        template<is_scalar_type_float_32bits F>
        consteval F mantissa_mask()
        {
            // 1位符号，8位指数，23位尾数
            return (~UINT32_C(0)) >> 9;
        }

        template<is_scalar_type_float_64bits F>
        consteval F mantissa_mask()
        {
            // 1位符号位，11位指数，52位尾数
            return (~UINT64_C(0)) >> 12;
        }
    } // namespace detail

    template<is_scalar_type S>
    using same_bits_uint_t = typename detail::uint_from_bytes<sizeof(S)>::type;

    template<typename S, int index>
    KSIMD_HEADER_GLOBAL_CONSTEXPR int InverseBitIndex = detail::inverse_bit_index_impl<S, index>();

    /**
     * @brief 0b11111111...
     */
    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR S OneBlock = std::bit_cast<S>(~static_cast<same_bits_uint_t<S>>(0));

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
        static_cast<same_bits_uint_t<S>>(1) << static_cast<same_bits_uint_t<S>>(InverseBitIndex<S, 0>)
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

    /**
     * @brief 尾数位全为1，其余位全为0
     */
    template<is_scalar_floating_point F>
    KSIMD_HEADER_GLOBAL_CONSTEXPR F MantissaMask = detail::mantissa_mask<F>();

    template<is_scalar_floating_point F>
    KSIMD_HEADER_GLOBAL_CONSTEXPR F Inf = std::numeric_limits<F>::infinity();

    template<is_scalar_floating_point F>
    KSIMD_HEADER_GLOBAL_CONSTEXPR F QNaN = std::numeric_limits<F>::quiet_NaN();

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

    template<is_scalar_floating_point F>
    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_NaN(const F f) noexcept
    {
        // 指数位均为1，尾数位 != 0
        using uint_t = same_bits_uint_t<F>;

        const uint_t bits = bitcast_to_uint(f);
        constexpr uint_t exp_mask = bitcast_to_uint(ExpMask<F>);
        constexpr uint_t mantissa_mask = bitcast_to_uint(MantissaMask<F>);
        return ((bits & exp_mask) == exp_mask) && (bits & mantissa_mask);
    }

    template<is_scalar_floating_point F>
    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_finite(const F f) noexcept
    {
        // 指数位不全为1
        using uint_t = same_bits_uint_t<F>;

        const uint_t bits = bitcast_to_uint(f);
        constexpr uint_t exp_mask = bitcast_to_uint(ExpMask<F>);
        return (bits & exp_mask) != exp_mask;
    }

    template<is_scalar_floating_point F>
    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_inf(const F f) noexcept
    {
        return (f == Inf<F>) || (f == -Inf<F>);
    }
}

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

        template<is_scalar_floating_point F>
        consteval F exp_mask()
        {
            if constexpr (std::is_same_v<float, F>)
            {
                constexpr uint32_t mask = UINT32_C(0b1111'1111) << 23;
                return std::bit_cast<F>(mask);
            }
            else if constexpr (std::is_same_v<double, F>)
            {
                constexpr uint64_t mask = UINT64_C(0b111'1111'1111) << 52;
                return std::bit_cast<F>(mask);
            }
            else
            {
                return 0;
            }
        }
    } // namespace detail

    template<is_scalar_type S>
    using same_bits_uint_t = typename detail::uint_from_bytes<sizeof(S)>::type;

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

    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_NaN(const float f) noexcept
    {
        // 指数位均为1，尾数位 != 0
        // 1位符号，8位指数，23位尾数
        const uint32_t bits = std::bit_cast<uint32_t>(f);
        constexpr uint32_t exp_mask = UINT32_C(0b1111'1111) << 23;
        constexpr uint32_t mantissa_mask = (~UINT32_C(0)) >> 9;
        return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0u);
    }

    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_NaN(const double f) noexcept
    {
        // 1位符号位，11位指数，52位尾数
        const uint64_t bits = std::bit_cast<uint64_t>(f);
        constexpr uint64_t exp_mask = UINT64_C(0b111'1111'1111) << 52;
        constexpr uint64_t mantissa_mask = (~UINT64_C(0)) >> 12;
        return ((bits & exp_mask) == exp_mask) && ((bits & mantissa_mask) != 0ull);
    }

    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_finite(const float f) noexcept
    {
        // 指数位不全为1
        const uint32_t bits = std::bit_cast<uint32_t>(f);
        constexpr uint32_t exp_mask = UINT32_C(0b1111'1111) << 23;
        return (bits & exp_mask) != exp_mask;
    }

    KSIMD_FORCE_INLINE KSIMD_FLATTEN constexpr bool is_finite(const double f) noexcept
    {
        const uint64_t bits = std::bit_cast<uint64_t>(f);
        constexpr uint64_t exp_mask = UINT64_C(0b111'1111'1111) << 52;
        return (bits & exp_mask) != exp_mask;
    }

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

    template<is_scalar_floating_point F>
    KSIMD_HEADER_GLOBAL_CONSTEXPR F Inf = std::numeric_limits<F>::infinity();

    template<is_scalar_floating_point F>
    KSIMD_HEADER_GLOBAL_CONSTEXPR F QNaN = std::numeric_limits<F>::quiet_NaN();
}

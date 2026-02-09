// do not use include guard

#include <cmath>
#include <cstring>
#include <type_traits>

#include "kSimd/IDE/IDE_hint.hpp"

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/traits.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_SCALAR_INTRINSIC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        // --- types ---
        template<is_scalar_type S>
        struct Batch
        {
            S v;
        };

        template<is_scalar_type S>
        struct Mask
        {
            same_bits_uint_t<S> m;
        };


        template<is_scalar_type S>
        struct op : OpHelper
        {
            using scalar_t = S;
            using batch_t = Batch<S>;
            using mask_t = Mask<S>;
            static constexpr size_t Alignment = alignof(S);
            static constexpr size_t Lanes = 1;

#pragma region memory
            KSIMD_API(batch_t) load(const scalar_t* mem) noexcept
            {
                return { *mem };
            }
            KSIMD_API(batch_t) loadu(const scalar_t* mem) noexcept
            {
                return { *mem };
            }
            KSIMD_API(batch_t) load_partial(const scalar_t* mem, size_t count) noexcept
            {
                return count > 0 ? batch_t{ *mem } : zero();
            }
            KSIMD_API(void) store(scalar_t* mem, batch_t v) noexcept
            {
                *mem = v.v;
            }
            KSIMD_API(void) storeu(scalar_t* mem, batch_t v) noexcept
            {
                *mem = v.v;
            }
            KSIMD_API(void) store_partial(scalar_t* mem, batch_t v, size_t count) noexcept
            {
                if (count > 0)
                    *mem = v.v;
            }
#pragma endregion

#pragma region initialization
            KSIMD_API(batch_t) undefined() noexcept
            {
                return {};
            }
            KSIMD_API(batch_t) zero() noexcept
            {
                return { static_cast<S>(0) };
            }
            KSIMD_API(batch_t) set(scalar_t x) noexcept
            {
                return { x };
            }
            KSIMD_API(batch_t) sequence() noexcept
            {
                return { static_cast<S>(0) };
            }
            KSIMD_API(batch_t) sequence(scalar_t base) noexcept
            {
                return { base };
            }
            KSIMD_API(batch_t) sequence(scalar_t base, scalar_t /*stride*/) noexcept
            {
                return { base };
            }
#pragma endregion

#pragma region math
            KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
            {
                return { lhs.v + rhs.v };
            }
            KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
            {
                return { lhs.v - rhs.v };
            }
            KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
            {
                return { lhs.v * rhs.v };
            }
            KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
            {
                return { lhs.v / rhs.v };
            }

            KSIMD_API(batch_t) one_div(batch_t v) noexcept
            {
                return { static_cast<S>(1) / v.v };
            }

            KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
            {
                return { a.v * b.v + c.v };
            }

            KSIMD_API(batch_t) sqrt(batch_t v) noexcept
            {
                return { std::sqrt(v.v) };
            }

            KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
            {
                return { static_cast<S>(1) / std::sqrt(v.v) };
            }

            template<RoundingMode mode>
            KSIMD_API(batch_t) round(batch_t v) noexcept
            {
                if constexpr (mode == RoundingMode::Up)
                    return { std::ceil(v.v) };
                else if constexpr (mode == RoundingMode::Down)
                    return { std::floor(v.v) };
                else if constexpr (mode == RoundingMode::Nearest)
                    return { std::nearbyint(v.v) };
                else if constexpr (mode == RoundingMode::ToZero)
                    return { std::trunc(v.v) };
                else /* if constexpr (mode == RoundingMode::Round) */
                    return { std::round(v.v) };
            }

            KSIMD_API(batch_t) abs(batch_t v) noexcept
            {
                if constexpr (is_scalar_floating_point<S>)
                    return { std::abs(v.v) };
                else
                    return { v.v < 0 ? -v.v : v.v };
            }

            KSIMD_API(batch_t) neg(batch_t v) noexcept
            {
                return { -v.v };
            }
            KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
            {
                return { ksimd::min(lhs.v, rhs.v) };
            }
            KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
            {
                return { ksimd::max(lhs.v, rhs.v) };
            }

            KSIMD_API(batch_t) bit_not(batch_t v) noexcept
            {
                auto bits = std::bit_cast<same_bits_uint_t<S>>(v.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(~bits)) };
            }

            KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
            {
                auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l & r)) };
            }

            KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
            {
                auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>((~l) & r)) };
            }

            KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
            {
                auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l | r)) };
            }

            KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
            {
                auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l ^ r)) };
            }

            KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
            {
                auto m = std::bit_cast<same_bits_uint_t<S>>(mask.v);
                auto av = std::bit_cast<same_bits_uint_t<S>>(a.v);
                auto bv = std::bit_cast<same_bits_uint_t<S>>(b.v);
                return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>((m & av) | ((~m) & bv))) };
            }
#pragma endregion

#pragma region mask logic
#if defined(KSIMD_IS_TESTING)
            KSIMD_API(void) test_store_mask(scalar_t* mem, mask_t mask) noexcept
            {
                std::memcpy(mem, &mask.m, sizeof(scalar_t));
            }
            KSIMD_API(mask_t) test_load_mask(const scalar_t* mem) noexcept
            {
                mask_t res;
                std::memcpy(&res.m, mem, sizeof(scalar_t));
                return res;
            }
#endif

            KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v == rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v != rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v > rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
            {
                return { (!(lhs.v > rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v >= rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (!(lhs.v >= rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v < rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
            {
                return { (!(lhs.v < rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (lhs.v <= rhs.v) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }
            KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
            {
                return { (!(lhs.v <= rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
            }

            KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
            {
                if constexpr (std::is_floating_point_v<S>)
                    return { (std::isnan(lhs.v) || std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
                return { ZeroBlock<S> };
            }

            KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
            {
                if constexpr (std::is_floating_point_v<S>)
                    return { (std::isnan(lhs.v) && std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
                return { ZeroBlock<S> };
            }

            KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
            {
                if constexpr (std::is_floating_point_v<S>)
                    return { (!std::isnan(lhs.v) && !std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
                return { OneBlock<S> };
            }

            KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
            {
                if constexpr (std::is_floating_point_v<S>)
                    return { (std::isfinite(lhs.v) || std::isfinite(rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
                return { OneBlock<S> };
            }

            KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
            {
                if constexpr (std::is_floating_point_v<S>)
                    return { (std::isfinite(lhs.v) && std::isfinite(rhs.v)) ? OneBlock<same_bits_uint_t<scalar_t>> : ZeroBlock<same_bits_uint_t<scalar_t>> };
                return { OneBlock<S> };
            }

            KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
            {
                using uint_t = same_bits_uint_t<S>;
                uint_t m = mask.m;
                uint_t au = std::bit_cast<uint_t>(a.v);
                uint_t bu = std::bit_cast<uint_t>(b.v);
                return { std::bit_cast<S>(static_cast<uint_t>((m & au) | ((~m) & bu))) };
            }
#pragma endregion

#pragma region reduce operation
            KSIMD_API(scalar_t) reduce_add(batch_t v) noexcept
            {
                return v.v;
            }
#pragma endregion
        };
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace ksimd

#undef KSIMD_API

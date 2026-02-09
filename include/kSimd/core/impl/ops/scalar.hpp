// do not use include guard

#include <cmath>
#include <cstring>
#include <type_traits>

#include "kSimd/IDE/IDE_hint.hpp"

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/traits.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

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

        namespace detail
        {
            template<is_scalar_type S>
            struct op_any_type_impl
            {
#pragma region memory
                KSIMD_API(Batch<S>) load(const S* mem) noexcept
                {
                    return { *mem };
                }
                KSIMD_API(Batch<S>) loadu(const S* mem) noexcept
                {
                    return { *mem };
                }
                KSIMD_API(Batch<S>) load_partial(const S* mem, size_t count) noexcept
                {
                    return count > 0 ? Batch<S>{ *mem } : zero();
                }
                KSIMD_API(void) store(S* mem, Batch<S> v) noexcept
                {
                    *mem = v.v;
                }
                KSIMD_API(void) storeu(S* mem, Batch<S> v) noexcept
                {
                    *mem = v.v;
                }
                KSIMD_API(void) store_partial(S* mem, Batch<S> v, size_t count) noexcept
                {
                    if (count > 0)
                        *mem = v.v;
                }
#pragma endregion

#pragma region initialization
                KSIMD_API(Batch<S>) undefined() noexcept
                {
                    return {};
                }
                KSIMD_API(Batch<S>) zero() noexcept
                {
                    return { static_cast<S>(0) };
                }
                KSIMD_API(Batch<S>) set(S x) noexcept
                {
                    return { x };
                }
                KSIMD_API(Batch<S>) sequence() noexcept
                {
                    return { static_cast<S>(0) };
                }
                KSIMD_API(Batch<S>) sequence(S base) noexcept
                {
                    return { base };
                }
                KSIMD_API(Batch<S>) sequence(S base, S /*stride*/) noexcept
                {
                    return { base };
                }
#pragma endregion

#pragma region math
                KSIMD_API(Batch<S>) add(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { lhs.v + rhs.v };
                }
                KSIMD_API(Batch<S>) sub(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { lhs.v - rhs.v };
                }
                KSIMD_API(Batch<S>) mul(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { lhs.v * rhs.v };
                }
                KSIMD_API(Batch<S>) div(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { lhs.v / rhs.v };
                }

                KSIMD_API(Batch<S>) mul_add(Batch<S> a, Batch<S> b, Batch<S> c) noexcept
                {
                    return { a.v * b.v + c.v };
                }
                KSIMD_API(Batch<S>) min(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ksimd::min(lhs.v, rhs.v) };
                }
                KSIMD_API(Batch<S>) max(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ksimd::max(lhs.v, rhs.v) };
                }

                KSIMD_API(Batch<S>) bit_not(Batch<S> v) noexcept
                {
                    auto bits = std::bit_cast<same_bits_uint_t<S>>(v.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(~bits)) };
                }

                KSIMD_API(Batch<S>) bit_and(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                    auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l & r)) };
                }

                KSIMD_API(Batch<S>) bit_and_not(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                    auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>((~l) & r)) };
                }

                KSIMD_API(Batch<S>) bit_or(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                    auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l | r)) };
                }

                KSIMD_API(Batch<S>) bit_xor(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    auto l = std::bit_cast<same_bits_uint_t<S>>(lhs.v);
                    auto r = std::bit_cast<same_bits_uint_t<S>>(rhs.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>(l ^ r)) };
                }

                KSIMD_API(Batch<S>) bit_if_then_else(Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
                {
                    auto m = std::bit_cast<same_bits_uint_t<S>>(_if.v);
                    auto av = std::bit_cast<same_bits_uint_t<S>>(_then.v);
                    auto bv = std::bit_cast<same_bits_uint_t<S>>(_else.v);
                    return { std::bit_cast<S>(static_cast<same_bits_uint_t<S>>((m & av) | ((~m) & bv))) };
                }
#pragma endregion

#pragma region mask logic
#if defined(KSIMD_IS_TESTING)
                KSIMD_API(void) test_store_mask(S* mem, Mask<S> mask) noexcept
                {
                    std::memcpy(mem, &mask.m, sizeof(S));
                }
                KSIMD_API(Mask<S>) test_load_mask(const S* mem) noexcept
                {
                    Mask<S> res;
                    std::memcpy(&res.m, mem, sizeof(S));
                    return res;
                }
#endif

                KSIMD_API(Mask<S>) equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v == rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }
                KSIMD_API(Mask<S>) not_equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v != rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }
                KSIMD_API(Mask<S>) greater(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v > rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v >= rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) less(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v < rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (lhs.v <= rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) mask_and(Mask<S> lhs, Mask<S> rhs) noexcept
                {
                    return { lhs.m & rhs.m };
                }

                KSIMD_API(Mask<S>) mask_or(Mask<S> lhs, Mask<S> rhs) noexcept
                {
                    return { lhs.m | rhs.m };
                }

                KSIMD_API(Mask<S>) mask_xor(Mask<S> lhs, Mask<S> rhs) noexcept
                {
                    return { lhs.m ^ rhs.m };
                }

                KSIMD_API(Mask<S>) mask_not(Mask<S> mask) noexcept
                {
                    return { ~mask.m };
                }

                KSIMD_API(Batch<S>) if_then_else(Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
                {
                    using uint_t = same_bits_uint_t<S>;
                    uint_t m = _if.m;
                    uint_t au = std::bit_cast<uint_t>(_then.v);
                    uint_t bu = std::bit_cast<uint_t>(_else.v);
                    return { std::bit_cast<S>(static_cast<uint_t>((m & au) | ((~m) & bu))) };
                }
#pragma endregion

#pragma region reduce operation
                KSIMD_API(S) reduce_add(Batch<S> v) noexcept
                {
                    return v.v;
                }
#pragma endregion
            };

            template<typename>
            struct op_signed_type_empty
            {};

            template<is_scalar_type S>
            struct op_signed_type_impl
            {
                KSIMD_API(Batch<S>) abs(Batch<S> v) noexcept
                {
                    if constexpr (std::is_floating_point_v<S>)
                        return { std::abs(v.v) };
                    else
                        return { v.v < 0 ? -v.v : v.v };
                }

                KSIMD_API(Batch<S>) neg(Batch<S> v) noexcept
                {
                    return { -v.v };
                }
            };

            template<typename>
            struct op_float32_float64_empty
            {};

            template<is_scalar_type_includes<float32, float64> S>
            struct op_float32_float64_impl
            {
                KSIMD_API(Batch<S>) one_div(Batch<S> v) noexcept
                {
                    return { static_cast<S>(1) / v.v };
                }

                KSIMD_API(Batch<S>) sqrt(Batch<S> v) noexcept
                {
                    return { std::sqrt(v.v) };
                }

                KSIMD_API(Batch<S>) rsqrt(Batch<S> v) noexcept
                {
                    return { static_cast<S>(1) / std::sqrt(v.v) };
                }

                template<OpHelper::RoundingMode mode>
                KSIMD_API(Batch<S>) round(Batch<S> v) noexcept
                {
                    if constexpr (mode == OpHelper::RoundingMode::Up)
                        return { std::ceil(v.v) };
                    else if constexpr (mode == OpHelper::RoundingMode::Down)
                        return { std::floor(v.v) };
                    else if constexpr (mode == OpHelper::RoundingMode::Nearest)
                        return { std::nearbyint(v.v) };
                    else if constexpr (mode == OpHelper::RoundingMode::ToZero)
                        return { std::trunc(v.v) };
                    else /* if constexpr (mode == RoundingMode::Round) */
                        return { std::round(v.v) };
                }

                KSIMD_API(Mask<S>) not_greater(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ~op_any_type_impl<S>::greater(lhs, rhs).m };
                }

                KSIMD_API(Mask<S>) not_greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ~op_any_type_impl<S>::greater_equal(lhs, rhs).m };
                }

                KSIMD_API(Mask<S>) not_less(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ~op_any_type_impl<S>::less(lhs, rhs).m };
                }

                KSIMD_API(Mask<S>) not_less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { ~op_any_type_impl<S>::less_equal(lhs, rhs).m };
                }

                KSIMD_API(Mask<S>) any_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (std::isnan(lhs.v) || std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                      : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) all_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (std::isnan(lhs.v) && std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                      : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) not_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (!std::isnan(lhs.v) && !std::isnan(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                        : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) any_finite(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (std::isfinite(lhs.v) || std::isfinite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                            : ZeroBlock<same_bits_uint_t<S>> };
                }

                KSIMD_API(Mask<S>) all_finite(Batch<S> lhs, Batch<S> rhs) noexcept
                {
                    return { (std::isfinite(lhs.v) && std::isfinite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                            : ZeroBlock<same_bits_uint_t<S>> };
                }
            };

            // selectors
            template<typename S, bool IsSigned = std::is_signed_v<S>>
            struct signed_type_impl_selector : op_signed_type_empty<S>
            {};

            template<typename S>
            struct signed_type_impl_selector<S, true> : op_signed_type_impl<S>
            {};

            template<typename S, bool IsFloat32_64 = is_scalar_type_includes<S, float32, float64>>
            struct float32_float64_impl_selector : op_float32_float64_empty<S>
            {};

            template<typename S>
            struct float32_float64_impl_selector<S, true> : op_float32_float64_impl<S>
            {};
        } // namespace detail


        template<is_scalar_type S>
        struct op
            : OpInfo<S, Batch<S>, Mask<S>, sizeof(S), alignof(S)>
            , OpHelper
            // any type
            , detail::op_any_type_impl<S>
            // signed type
            , detail::signed_type_impl_selector<S>
            // float32, float64
            , detail::float32_float64_impl_selector<S>
        {
            using scalar_t = S;
            using batch_t = Batch<S>;
            using mask_t = Mask<S>;
            static constexpr size_t Alignment = alignof(S);
            static constexpr size_t Lanes = 1;
        };
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace ksimd

#undef KSIMD_API

#include "operators.inl"

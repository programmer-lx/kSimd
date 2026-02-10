// do not use include guard

#include "op_helpers.hpp"
#include "op_helpers.hpp"


#include <cmath>
#include <cstring>
#include <type_traits>

#include "kSimd/IDE/IDE_hint.hpp"

#include "op_helpers.hpp"
#include "kSimd/core/impl/func_attr.hpp"
#include "kSimd/core/impl/traits.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
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

            template<OpHelper::FloatMinMaxOption option = OpHelper::FloatMinMaxOption::Native>
            KSIMD_API(Batch<S>) min(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                if constexpr (option == OpHelper::FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                {
                    return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? QNaN<S> : ksimd::min(lhs.v, rhs.v) };
                }
                else
                {
                    return { ksimd::min(lhs.v, rhs.v) };
                }
            }

            template<OpHelper::FloatMinMaxOption option = OpHelper::FloatMinMaxOption::Native>
            KSIMD_API(Batch<S>) max(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                if constexpr (option == OpHelper::FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                {
                    return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? QNaN<S> : ksimd::max(lhs.v, rhs.v) };
                }
                else
                {
                    return { ksimd::max(lhs.v, rhs.v) };
                }
            }
#pragma endregion

#pragma region bit logic
            KSIMD_API(Batch<S>) bit_not(Batch<S> v) noexcept
            {
                auto bits = ksimd::bitcast_to_uint<S>(v.v);
                return { std::bit_cast<S>(~bits) };
            }

            KSIMD_API(Batch<S>) bit_and(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                auto l = ksimd::bitcast_to_uint<S>(lhs.v);
                auto r = ksimd::bitcast_to_uint<S>(rhs.v);
                return { std::bit_cast<S>(l & r) };
            }

            KSIMD_API(Batch<S>) bit_and_not(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                auto l = ksimd::bitcast_to_uint<S>(lhs.v);
                auto r = ksimd::bitcast_to_uint<S>(rhs.v);
                return { std::bit_cast<S>((~l) & r) };
            }

            KSIMD_API(Batch<S>) bit_or(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                auto l = ksimd::bitcast_to_uint<S>(lhs.v);
                auto r = ksimd::bitcast_to_uint<S>(rhs.v);
                return { std::bit_cast<S>(l | r) };
            }

            KSIMD_API(Batch<S>) bit_xor(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                auto l = ksimd::bitcast_to_uint<S>(lhs.v);
                auto r = ksimd::bitcast_to_uint<S>(rhs.v);
                return { std::bit_cast<S>(l ^ r) };
            }

            KSIMD_API(Batch<S>) bit_if_then_else(Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
            {
                auto _if_v = ksimd::bitcast_to_uint<S>(_if.v);
                auto _then_v = ksimd::bitcast_to_uint<S>(_then.v);
                auto _else_v = ksimd::bitcast_to_uint<S>(_else.v);
                return { std::bit_cast<S>((_if_v & _then_v) | ((~_if_v) & _else_v)) };
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
                uint_t _if_v = _if.m;
                uint_t _then_v = std::bit_cast<uint_t>(_then.v);
                uint_t _else_v = std::bit_cast<uint_t>(_else.v);
                return { std::bit_cast<S>((_if_v & _then_v) | ((~_if_v) & _else_v)) };
            }
#pragma endregion

#pragma region reduce operation
            KSIMD_API(S) reduce_add(Batch<S> v) noexcept
            {
                return v.v;
            }

            KSIMD_API(S) reduce_mul(Batch<S> v) noexcept
            {
                return v.v;
            }

            template<OpHelper::FloatMinMaxOption = OpHelper::FloatMinMaxOption::Native>
            KSIMD_API(S) reduce_min(Batch<S> v) noexcept
            {
                return v.v;
            }

            template<OpHelper::FloatMinMaxOption = OpHelper::FloatMinMaxOption::Native>
            KSIMD_API(S) reduce_max(Batch<S> v) noexcept
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
                return { std::abs(v.v) };
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
            KSIMD_API(Batch<S>) sqrt(Batch<S> v) noexcept
            {
                return { std::sqrt(v.v) };
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
                return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                        : ZeroBlock<same_bits_uint_t<S>> };
            }

            KSIMD_API(Mask<S>) all_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                return { (ksimd::is_NaN(lhs.v) && ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                        : ZeroBlock<same_bits_uint_t<S>> };
            }

            KSIMD_API(Mask<S>) not_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                return { (!ksimd::is_NaN(lhs.v) && !ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                          : ZeroBlock<same_bits_uint_t<S>> };
            }

            KSIMD_API(Mask<S>) any_finite(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                return { (ksimd::is_finite(lhs.v) || ksimd::is_finite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                              : ZeroBlock<same_bits_uint_t<S>> };
            }

            KSIMD_API(Mask<S>) all_finite(Batch<S> lhs, Batch<S> rhs) noexcept
            {
                return { (ksimd::is_finite(lhs.v) && ksimd::is_finite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                              : ZeroBlock<same_bits_uint_t<S>> };
            }
        };

        struct op_float32_empty
        {};

        struct op_float32_impl
        {
            KSIMD_API(Batch<float32>) rcp(Batch<float32> v) noexcept
            {
                return { 1.0f / v.v };
            }

            KSIMD_API(Batch<float32>) rsqrt(Batch<float32> v) noexcept
            {
                return { static_cast<float32>(1) / std::sqrt(v.v) };
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

        template<typename S, bool IsFloat32 = std::is_same_v<S, float32>>
        struct float32_impl_selector : op_float32_empty
        {};

        template<typename S>
        struct float32_impl_selector<S, true> : op_float32_impl
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

        // float32 only
        , detail::float32_impl_selector<S>
    {};
} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

#include "operators.inl"

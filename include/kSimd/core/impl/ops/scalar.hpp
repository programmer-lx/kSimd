// do not use include guard

// #include "kSimd/IDE/IDE_hint.hpp"

#include <cmath>
#include <cstring>

#include <utility> // index_sequence

#include "op_helpers.hpp"
#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- types ---
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
#pragma endregion

#pragma region--- constants ---
    template<is_scalar_type>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Lanes = 1;

    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignof(S);
#pragma endregion

#pragma region--- any type ---
    template<is_scalar_type S>
    KSIMD_API(Batch<S>) load(const S* mem) noexcept
    {
        return { *mem };
    }

    template<is_scalar_type S>
    KSIMD_API(void) store(S* mem, Batch<S> v) noexcept
    {
        *mem = v.v;
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu(const S* mem) noexcept
    {
        return { *mem };
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu(S* mem, Batch<S> v) noexcept
    {
        *mem = v.v;
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu_partial(const S* mem, size_t count) noexcept
    {
        return count > 0 ? Batch<S>{ *mem } : Batch<S>{ static_cast<S>(0) };
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu_partial(S* mem, Batch<S> v, size_t count) noexcept
    {
        if (count > 0)
            *mem = v.v;
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) undefined() noexcept
    {
        return {};
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) zero() noexcept
    {
        return { static_cast<S>(0) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) set(S x) noexcept
    {
        return { x };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence() noexcept
    {
        return { static_cast<S>(0) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(S base) noexcept
    {
        return { base };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(S base, S /*stride*/) noexcept
    {
        return { base };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) add(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { lhs.v + rhs.v };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sub(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { lhs.v - rhs.v };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { lhs.v * rhs.v };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) div(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { lhs.v / rhs.v };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul_add(Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        return { a.v * b.v + c.v };
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) min(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
        {
            return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? QNaN<S> : ksimd::min(lhs.v, rhs.v) };
        }
        else
        {
            return { ksimd::min(lhs.v, rhs.v) };
        }
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) max(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
        {
            return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? QNaN<S> : ksimd::max(lhs.v, rhs.v) };
        }
        else
        {
            return { ksimd::max(lhs.v, rhs.v) };
        }
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_not(Batch<S> v) noexcept
    {
        auto bits = ksimd::bitcast_to_uint<S>(v.v);
        return { std::bit_cast<S>(~bits) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        auto l = ksimd::bitcast_to_uint<S>(lhs.v);
        auto r = ksimd::bitcast_to_uint<S>(rhs.v);
        return { std::bit_cast<S>(l & r) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and_not(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        auto l = ksimd::bitcast_to_uint<S>(lhs.v);
        auto r = ksimd::bitcast_to_uint<S>(rhs.v);
        return { std::bit_cast<S>((~l) & r) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_or(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        auto l = ksimd::bitcast_to_uint<S>(lhs.v);
        auto r = ksimd::bitcast_to_uint<S>(rhs.v);
        return { std::bit_cast<S>(l | r) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_xor(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        auto l = ksimd::bitcast_to_uint<S>(lhs.v);
        auto r = ksimd::bitcast_to_uint<S>(rhs.v);
        return { std::bit_cast<S>(l ^ r) };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_if_then_else(Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        auto _if_v = ksimd::bitcast_to_uint<S>(_if.v);
        auto _then_v = ksimd::bitcast_to_uint<S>(_then.v);
        auto _else_v = ksimd::bitcast_to_uint<S>(_else.v);
        return { std::bit_cast<S>((_if_v & _then_v) | ((~_if_v) & _else_v)) };
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type S>
    KSIMD_API(void) test_store_mask(S* mem, Mask<S> mask) noexcept
    {
        std::memcpy(mem, &mask.m, sizeof(S));
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) test_load_mask(const S* mem) noexcept
    {
        Mask<S> res;
        std::memcpy(&res.m, mem, sizeof(S));
        return res;
    }
#endif

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v == rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) not_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v != rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v > rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v >= rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v < rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (lhs.v <= rhs.v) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_and(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { lhs.m & rhs.m };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_or(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { lhs.m | rhs.m };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_xor(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return { lhs.m ^ rhs.m };
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_not(Mask<S> mask) noexcept
    {
        return { ~mask.m };
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) if_then_else(Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        using uint_t = same_bits_uint_t<S>;

        uint_t _if_v = _if.m;
        uint_t _then_v = bitcast_to_uint(_then.v);
        uint_t _else_v = bitcast_to_uint(_else.v);
        return { std::bit_cast<S>((_if_v & _then_v) | ((~_if_v) & _else_v)) };
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_add(Batch<S> v) noexcept
    {
        return v.v;
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_mul(Batch<S> v) noexcept
    {
        return v.v;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_min(Batch<S> v) noexcept
    {
        return v.v;
    }

    template<FloatMinMaxOption = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_max(Batch<S> v) noexcept
    {
        return v.v;
    }
#pragma endregion

#pragma region--- signed ---
    template<is_scalar_signed S>
    KSIMD_API(Batch<S>) abs(Batch<S> v) noexcept
    {
        return { std::abs(v.v) };
    }

    template<is_scalar_signed S>
    KSIMD_API(Batch<S>) neg(Batch<S> v) noexcept
    {
        return { -v.v };
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) sqrt(Batch<S> v) noexcept
    {
        return { std::sqrt(v.v) };
    }

    template<RoundingMode mode, is_scalar_floating_point S>
    KSIMD_API(Batch<S>) round(Batch<S> v) noexcept
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

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { ~greater(lhs, rhs).m };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { ~greater_equal(lhs, rhs).m };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { ~less(lhs, rhs).m };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { ~less_equal(lhs, rhs).m };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (ksimd::is_NaN(lhs.v) || ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (ksimd::is_NaN(lhs.v) && ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (!ksimd::is_NaN(lhs.v) && !ksimd::is_NaN(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                  : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (ksimd::is_finite(lhs.v) || ksimd::is_finite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                      : ZeroBlock<same_bits_uint_t<S>> };
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return { (ksimd::is_finite(lhs.v) && ksimd::is_finite(rhs.v)) ? OneBlock<same_bits_uint_t<S>>
                                                                      : ZeroBlock<same_bits_uint_t<S>> };
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Batch<S> v) noexcept
    {
        return { static_cast<S>(1) / v.v };
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Batch<S> v) noexcept
    {
        return { static_cast<S>(1) / std::sqrt(v.v) };
    }
#pragma endregion

} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

#include "operators.inl"

// do not use include guard

// #include "kSimd/IDE/IDE_hint.hpp"

#include <cmath> // sqrt
#include <cstring> // memcpy, memset

#include <utility> // index_sequence

#include "op_helpers.hpp"
#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- traits ---
    template<is_scalar_type S>
    struct Traits
    {
        using _scalar_type = S;
        static constexpr size_t _lanes = vec_size::ScalarSim / sizeof(S);
    };
    
    template<is_scalar_type S>
    constexpr size_t lanes(Traits<S>) noexcept
    {
        return Traits<S>::_lanes;
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignof(std::max_align_t);
#pragma endregion

#pragma region--- types ---
    template<is_scalar_type S>
    struct Batch
    {
        S v[lanes(Traits<S>{})];
    };

    template<is_scalar_type S>
    struct Mask
    {
        same_bits_uint_t<S> m[lanes(Traits<S>{})];
    };
#pragma endregion

#pragma region--- any type ---
    template<is_scalar_type S>
    KSIMD_API(Batch<S>) load(Traits<S>, const S* mem) noexcept
    {
        Batch<S> res{};
        std::memcpy(res.v, mem, sizeof(S) * lanes(Traits<S>{}));
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(void) store(Traits<S>, S* mem, Batch<S> v) noexcept
    {
        std::memcpy(mem, v.v, sizeof(S) * lanes(Traits<S>{}));
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu(Traits<S> t, const S* mem) noexcept
    {
        return load(t, mem);
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu(Traits<S> t, S* mem, Batch<S> v) noexcept
    {
        store(t, mem, v);
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu_partial(Traits<S>, const S* mem, size_t count) noexcept
    {
        Batch<S> res = { static_cast<S>(0) };
        std::memcpy(res.v, mem, sizeof(S) * ksimd::min(count, lanes(Traits<S>{})));
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu_partial(Traits<S>, S* mem, Batch<S> v, size_t count) noexcept
    {
        std::memcpy(mem, v.v, sizeof(S) * ksimd::min(count, lanes(Traits<S>{})));
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) undefined(Traits<S>) noexcept
    {
        return {};
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) zero(Traits<S>) noexcept
    {
        Batch<S> res;
        std::memset(res.v, 0x00, sizeof(S) * lanes(Traits<S>{}));
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) set(Traits<S>, S x) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { ((void)I, x)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(Traits<S>) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(I)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(Traits<S>, S base) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(base + I)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(Traits<S>, S base, S stride) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(base + static_cast<S>(I) * stride)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) add(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] + rhs.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sub(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] - rhs.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] * rhs.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul_add(Traits<S>, Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(a.v[I] * b.v[I] + c.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) min(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            auto element_min = [&](size_t i)
            {
                if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                    return (ksimd::is_NaN(lhs.v[i]) || ksimd::is_NaN(rhs.v[i])) ? QNaN<S>
                                                                                : ksimd::min(lhs.v[i], rhs.v[i]);
                else
                    return ksimd::min(lhs.v[i], rhs.v[i]);
            };
            return { element_min(I)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) max(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            auto element_max = [&](size_t i)
            {
                if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                    return (ksimd::is_NaN(lhs.v[i]) || ksimd::is_NaN(rhs.v[i])) ? QNaN<S>
                                                                                : ksimd::max(lhs.v[i], rhs.v[i]);
                else
                    return ksimd::max(lhs.v[i], rhs.v[i]);
            };
            return { element_max(I)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_not(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(~ksimd::bitcast_to_uint<S>(v.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) & ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and_not(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>((~ksimd::bitcast_to_uint<S>(lhs.v[I])) &
                                      ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_or(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) | ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_xor(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) ^ ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_if_then_else(Traits<S>, Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(
                    (ksimd::bitcast_to_uint<S>(_if.v[I]) & ksimd::bitcast_to_uint<S>(_then.v[I])) |
                    ((~ksimd::bitcast_to_uint<S>(_if.v[I])) & ksimd::bitcast_to_uint<S>(_else.v[I])))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type S>
    KSIMD_API(void) test_store_mask(Traits<S>, S* mem, Mask<S> mask) noexcept
    {
        std::memcpy(mem, mask.m, sizeof(S) * lanes(Traits<S>{}));
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) test_load_mask(Traits<S>, const S* mem) noexcept
    {
        Mask<S> res;
        std::memcpy(res.m, mem, sizeof(S) * lanes(Traits<S>{}));
        return res;
    }
#endif

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] == rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) not_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] != rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] > rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] >= rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] < rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less_equal(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] <= rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_and(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] & rhs.m[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_or(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] | rhs.m[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_xor(Traits<S>, Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] ^ rhs.m[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_not(Traits<S>, Mask<S> mask) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(~mask.m[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) if_then_else(Traits<S>, Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            auto select = [&](size_t i)
            {
                using uint_t = same_bits_uint_t<S>;
                uint_t cond = _if.m[i];
                uint_t t = bitcast_to_uint(_then.v[i]);
                uint_t e = bitcast_to_uint(_else.v[i]);
                return std::bit_cast<S>((cond & t) | ((~cond) & e));
            };
            return { select(I)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_add(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> S
        {
            return (v.v[I] + ...);
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_mul(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> S
        {
            return (v.v[I] * ...);
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_min(Traits<S>, Batch<S> v) noexcept
    {
        S res = v.v[0];
        for (size_t i = 1; i < lanes(Traits<S>{}); ++i)
        {
            if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                res = (ksimd::is_NaN(res) || ksimd::is_NaN(v.v[i])) ? QNaN<S> : ksimd::min(res, v.v[i]);
            else
                res = ksimd::min(res, v.v[i]);
        }
        return res;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_max(Traits<S>, Batch<S> v) noexcept
    {
        S res = v.v[0];
        for (size_t i = 1; i < lanes(Traits<S>{}); ++i)
        {
            if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                res = (ksimd::is_NaN(res) || ksimd::is_NaN(v.v[i])) ? QNaN<S> : ksimd::max(res, v.v[i]);
            else
                res = ksimd::max(res, v.v[i]);
        }
        return res;
    }
#pragma endregion

#pragma region--- signed ---
    template<is_scalar_signed S>
    KSIMD_API(Batch<S>) abs(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::abs(v.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_signed S>
    KSIMD_API(Batch<S>) neg(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(-v.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) div(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (lhs.v[I] / rhs.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) sqrt(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::sqrt(v.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<RoundingMode mode, is_scalar_floating_point S>
    KSIMD_API(Batch<S>) round(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            auto round_one = [&](S val)
            {
                if constexpr (mode == RoundingMode::Up)
                    return std::ceil(val);
                else if constexpr (mode == RoundingMode::Down)
                    return std::floor(val);
                else if constexpr (mode == RoundingMode::Nearest)
                    return std::nearbyint(val);
                else if constexpr (mode == RoundingMode::ToZero)
                    return std::trunc(val);
                else
                    return std::round(val);
            };
            return { round_one(v.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater(Traits<S> t, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(t, greater(t, lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater_equal(Traits<S> t, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(t, greater_equal(t, lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less(Traits<S> t, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(t, less(t, lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less_equal(Traits<S> t, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(t, less_equal(t, lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) || ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                           : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) && ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                           : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_NaN(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((!ksimd::is_NaN(lhs.v[I]) && !ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                             : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_finite(lhs.v[I]) || ksimd::is_finite(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                                 : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_finite(Traits<S>, Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_finite(lhs.v[I]) && ksimd::is_finite(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                                 : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (static_cast<S>(1.0f) / v.v[I])... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Traits<S>, Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (ksimd::rsqrt(v.v[I]))... };
        }(std::make_index_sequence<lanes(Traits<S>{})>{});
    }
#pragma endregion

} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

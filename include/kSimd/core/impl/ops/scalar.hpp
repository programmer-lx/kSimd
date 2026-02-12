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

#pragma region--- constants ---
    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Lanes = vec_size::ScalarSim / sizeof(S);

    template<is_scalar_type S>
    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignof(S);
#pragma endregion

#pragma region--- types ---
    template<is_scalar_type S>
    struct Batch
    {
        S v[Lanes<S>];
    };

    template<is_scalar_type S>
    struct Mask
    {
        same_bits_uint_t<S> m[Lanes<S>];
    };
#pragma endregion

#pragma region--- any type ---
    template<is_scalar_type S>
    KSIMD_API(Batch<S>) load(const S* mem) noexcept
    {
        Batch<S> res{};
        std::memcpy(res.v, mem, sizeof(S) * Lanes<S>);
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(void) store(S* mem, Batch<S> v) noexcept
    {
        std::memcpy(mem, v.v, sizeof(S) * Lanes<S>);
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu(const S* mem) noexcept
    {
        Batch<S> res{};
        std::memcpy(res.v, mem, sizeof(S) * Lanes<S>);
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu(S* mem, Batch<S> v) noexcept
    {
        std::memcpy(mem, v.v, sizeof(S) * Lanes<S>);
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) loadu_partial(const S* mem, size_t count) noexcept
    {
        Batch<S> res = { static_cast<S>(0) };
        std::memcpy(res.v, mem, sizeof(S) * ksimd::min(count, Lanes<S>));
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(void) storeu_partial(S* mem, Batch<S> v, size_t count) noexcept
    {
        std::memcpy(mem, v.v, sizeof(S) * ksimd::min(count, Lanes<S>));
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) undefined() noexcept
    {
        return {};
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) zero() noexcept
    {
        Batch<S> res = { static_cast<S>(0) };
        return res;
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) set(S x) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { ((void)I, x)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence() noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(I)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(S base) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(base + I)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sequence(S base, S stride) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(base + static_cast<S>(I) * stride)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) add(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] + rhs.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) sub(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] - rhs.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(lhs.v[I] * rhs.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) mul_add(Batch<S> a, Batch<S> b, Batch<S> c) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(a.v[I] * b.v[I] + c.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) min(Batch<S> lhs, Batch<S> rhs) noexcept
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
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) max(Batch<S> lhs, Batch<S> rhs) noexcept
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
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_not(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(~ksimd::bitcast_to_uint<S>(v.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) & ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_and_not(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>((~ksimd::bitcast_to_uint<S>(lhs.v[I])) &
                                      ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_or(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) | ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_xor(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(ksimd::bitcast_to_uint<S>(lhs.v[I]) ^ ksimd::bitcast_to_uint<S>(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) bit_if_then_else(Batch<S> _if, Batch<S> _then, Batch<S> _else) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::bit_cast<S>(
                    (ksimd::bitcast_to_uint<S>(_if.v[I]) & ksimd::bitcast_to_uint<S>(_then.v[I])) |
                    ((~ksimd::bitcast_to_uint<S>(_if.v[I])) & ksimd::bitcast_to_uint<S>(_else.v[I])))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

#if defined(KSIMD_IS_TESTING)
    template<is_scalar_type S>
    KSIMD_API(void) test_store_mask(S* mem, Mask<S> mask) noexcept
    {
        [&]<size_t... I>(std::index_sequence<I...>)
        {
            ((std::memcpy(mem + I, &mask.m[I], sizeof(S))), ...);
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) test_load_mask(const S* mem) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            auto load_one = [&](size_t i)
            {
                same_bits_uint_t<S> m;
                std::memcpy(&m, mem + i, sizeof(S));
                return m;
            };
            return { load_one(I)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }
#endif

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] == rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) not_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] != rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] > rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] >= rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] < rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((lhs.v[I] <= rhs.v[I]) ? OneBlock<same_bits_uint_t<S>> : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_and(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] & rhs.m[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_or(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] | rhs.m[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_xor(Mask<S> lhs, Mask<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(lhs.m[I] ^ rhs.m[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Mask<S>) mask_not(Mask<S> mask) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { static_cast<same_bits_uint_t<S>>(~mask.m[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(Batch<S>) if_then_else(Mask<S> _if, Batch<S> _then, Batch<S> _else) noexcept
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
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_add(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> S
        {
            return (v.v[I] + ...);
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type S>
    KSIMD_API(S) reduce_mul(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> S
        {
            return (v.v[I] * ...);
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_min(Batch<S> v) noexcept
    {
        S res = v.v[0];
        for (size_t i = 1; i < Lanes<S>; ++i)
        {
            if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<S>)
                res = (ksimd::is_NaN(res) || ksimd::is_NaN(v.v[i])) ? QNaN<S> : ksimd::min(res, v.v[i]);
            else
                res = ksimd::min(res, v.v[i]);
        }
        return res;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(S) reduce_max(Batch<S> v) noexcept
    {
        S res = v.v[0];
        for (size_t i = 1; i < Lanes<S>; ++i)
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
    KSIMD_API(Batch<S>) abs(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::abs(v.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_signed S>
    KSIMD_API(Batch<S>) neg(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { static_cast<S>(-v.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }
#pragma endregion

#pragma region--- floating point ---
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) div(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (lhs.v[I] / rhs.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) sqrt(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { std::sqrt(v.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<RoundingMode mode, is_scalar_floating_point S>
    KSIMD_API(Batch<S>) round(Batch<S> v) noexcept
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
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(greater(lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_greater_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(greater_equal(lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(less(lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_less_equal(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return mask_not(less_equal(lhs, rhs));
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) || ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                           : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) && ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                           : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) not_NaN(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((!ksimd::is_NaN(lhs.v[I]) && !ksimd::is_NaN(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                             : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) any_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_finite(lhs.v[I]) || ksimd::is_finite(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                                 : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_floating_point S>
    KSIMD_API(Mask<S>) all_finite(Batch<S> lhs, Batch<S> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<S>
        {
            return { ((ksimd::is_finite(lhs.v[I]) && ksimd::is_finite(rhs.v[I])) ? OneBlock<same_bits_uint_t<S>>
                                                                                 : ZeroBlock<same_bits_uint_t<S>>)... };
        }(std::make_index_sequence<Lanes<S>>{});
    }
#pragma endregion

#pragma region--- float32 only ---
    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rcp(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (static_cast<S>(1.0f) / v.v[I])... };
        }(std::make_index_sequence<Lanes<S>>{});
    }

    template<is_scalar_type_float_32bits S>
    KSIMD_API(Batch<S>) rsqrt(Batch<S> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S>
        {
            return { (ksimd::rsqrt(v.v[I]))... };
        }(std::make_index_sequence<Lanes<S>>{});
    }
#pragma endregion

} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

#include "operators.inl"

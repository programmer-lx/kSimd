// do not use include guard

#include <cmath>
#include <cstring>
#include <cstddef>

#include <utility>

#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{

#pragma region--- constants ---
    template<is_tag_full_and_fixed128 Tag>
    constexpr size_t lanes(Tag) noexcept
    {
        return vec_size::Scalar128 / sizeof(tag_scalar_t<Tag>);
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Scalar128;
#pragma endregion

#pragma region--- types ---
    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    struct Batch
    {
        tag_scalar_t<Tag> v[lanes(Tag{})];
    };

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    struct Mask
    {
        same_bits_uint_t<tag_scalar_t<Tag>> m[lanes(Tag{})];
    };
#pragma endregion

#pragma region--- any type ---
    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) load(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        Batch<Tag> res{};
        std::memcpy(res.v, mem, sizeof(tag_scalar_t<Tag>) * lanes(Tag{}));
        return res;
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) store(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        std::memcpy(mem, v.v, sizeof(tag_scalar_t<Tag>) * lanes(Tag{}));
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) loadu(Tag t, const tag_scalar_t<Tag>* mem) noexcept
    {
        return load(t, mem);
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) storeu(Tag t, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept
    {
        store(t, mem, v);
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) loadu_partial(Tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept
    {
        Batch<Tag> res = { static_cast<tag_scalar_t<Tag>>(0) };
        std::memcpy(res.v, mem, sizeof(tag_scalar_t<Tag>) * ksimd::min(count, lanes(Tag{})));
        return res;
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) storeu_partial(Tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept
    {
        std::memcpy(mem, v.v, sizeof(tag_scalar_t<Tag>) * ksimd::min(count, lanes(Tag{})));
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) undefined(Tag) noexcept
    {
        return {};
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) zero(Tag) noexcept
    {
        Batch<Tag> res;
        std::memset(res.v, 0x00, sizeof(tag_scalar_t<Tag>) * lanes(Tag{}));
        return res;
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) set(Tag, tag_scalar_t<Tag> x) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { ((void)I, x)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(I)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(base + I)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sequence(Tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(base + static_cast<tag_scalar_t<Tag>>(I) * stride)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) add(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(lhs.v[I] + rhs.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sub(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(lhs.v[I] - rhs.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) mul(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(lhs.v[I] * rhs.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) mul_add(Tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(a.v[I] * b.v[I] + c.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag Tag>
    KSIMD_API(Batch<Tag>) min(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            auto element_min = [&](size_t i)
            {
                if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<tag_scalar_t<Tag>>)
                    return (ksimd::is_NaN(lhs.v[i]) || ksimd::is_NaN(rhs.v[i])) ? QNaN<tag_scalar_t<Tag>>
                                                                                : ksimd::min(lhs.v[i], rhs.v[i]);
                else
                    return ksimd::min(lhs.v[i], rhs.v[i]);
            };
            return { element_min(I)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag Tag>
    KSIMD_API(Batch<Tag>) max(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            auto element_max = [&](size_t i)
            {
                if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<tag_scalar_t<Tag>>)
                    return (ksimd::is_NaN(lhs.v[i]) || ksimd::is_NaN(rhs.v[i])) ? QNaN<tag_scalar_t<Tag>>
                                                                                : ksimd::max(lhs.v[i], rhs.v[i]);
                else
                    return ksimd::max(lhs.v[i], rhs.v[i]);
            };
            return { element_max(I)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_not(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>(~ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(v.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>(ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(lhs.v[I]) &
                                                      ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_and_not(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>((~ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(lhs.v[I])) &
                                                      ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_or(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>(ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(lhs.v[I]) |
                                                      ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_xor(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>(ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(lhs.v[I]) ^
                                                      ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(rhs.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) bit_if_then_else(Tag, Batch<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::bit_cast<tag_scalar_t<Tag>>((ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(_if.v[I]) &
                                                       ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(_then.v[I])) |
                                                      ((~ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(_if.v[I])) &
                                                       ksimd::bitcast_to_uint<tag_scalar_t<Tag>>(_else.v[I])))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

#if defined(KSIMD_IS_TESTING)
    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(void) test_store_mask(Tag, tag_scalar_t<Tag>* mem, Mask<Tag> mask) noexcept
    {
        std::memcpy(mem, mask.m, sizeof(tag_scalar_t<Tag>) * lanes(Tag{}));
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) test_load_mask(Tag, const tag_scalar_t<Tag>* mem) noexcept
    {
        Mask<Tag> res;
        std::memcpy(res.m, mem, sizeof(tag_scalar_t<Tag>) * lanes(Tag{}));
        return res;
    }
#endif

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] == rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                             : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] != rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                             : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) greater(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] > rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                            : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) greater_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] >= rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                             : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) less(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] < rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                            : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) less_equal(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((lhs.v[I] <= rhs.v[I]) ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                                             : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_and(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { static_cast<same_bits_uint_t<tag_scalar_t<Tag>>>(lhs.m[I] & rhs.m[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_or(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { static_cast<same_bits_uint_t<tag_scalar_t<Tag>>>(lhs.m[I] | rhs.m[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_xor(Tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { static_cast<same_bits_uint_t<tag_scalar_t<Tag>>>(lhs.m[I] ^ rhs.m[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) mask_not(Tag, Mask<Tag> mask) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { static_cast<same_bits_uint_t<tag_scalar_t<Tag>>>(~mask.m[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) if_then_else(Tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            auto select = [&](size_t i)
            {
                using uint_t = same_bits_uint_t<tag_scalar_t<Tag>>;
                uint_t cond = _if.m[i];
                uint_t t = bitcast_to_uint(_then.v[i]);
                uint_t e = bitcast_to_uint(_else.v[i]);
                return std::bit_cast<tag_scalar_t<Tag>>((cond & t) | ((~cond) & e));
            };
            return { select(I)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_add(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> tag_scalar_t<Tag>
        {
            return (v.v[I] + ...);
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_full_and_fixed128<Tag>)
    KSIMD_API(tag_scalar_t<Tag>) reduce_mul(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> tag_scalar_t<Tag>
        {
            return (v.v[I] * ...);
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag Tag>
    KSIMD_API(tag_scalar_t<Tag>) reduce_min(Tag, Batch<Tag> v) noexcept
    {
        tag_scalar_t<Tag> res = v.v[0];
        for (size_t i = 1; i < lanes(Tag{}); ++i)
        {
            if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<tag_scalar_t<Tag>>)
                res = (ksimd::is_NaN(res) || ksimd::is_NaN(v.v[i])) ? QNaN<tag_scalar_t<Tag>> : ksimd::min(res, v.v[i]);
            else
                res = ksimd::min(res, v.v[i]);
        }
        return res;
    }

    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_tag Tag>
    KSIMD_API(tag_scalar_t<Tag>) reduce_max(Tag, Batch<Tag> v) noexcept
    {
        tag_scalar_t<Tag> res = v.v[0];
        for (size_t i = 1; i < lanes(Tag{}); ++i)
        {
            if constexpr (option == FloatMinMaxOption::CheckNaN && is_scalar_floating_point<tag_scalar_t<Tag>>)
                res = (ksimd::is_NaN(res) || ksimd::is_NaN(v.v[i])) ? QNaN<tag_scalar_t<Tag>> : ksimd::max(res, v.v[i]);
            else
                res = ksimd::max(res, v.v[i]);
        }
        return res;
    }
#pragma endregion

#pragma region--- signed ---
    template<typename Tag>
        requires (is_tag_signed<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) abs(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::abs(v.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_signed<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) neg(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { static_cast<tag_scalar_t<Tag>>(-v.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }
#pragma endregion

#pragma region--- floating point ---
    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) div(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { (lhs.v[I] / rhs.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) sqrt(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { std::sqrt(v.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<RoundingMode mode, typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) round(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            auto round_one = [&](tag_scalar_t<Tag> val)
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
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, greater(t, lhs, rhs));
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_greater_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, greater_equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_less(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, less(t, lhs, rhs));
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_less_equal(Tag t, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return mask_not(t, less_equal(t, lhs, rhs));
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) any_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) || ksimd::is_NaN(rhs.v[I]))
                              ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                              : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) all_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((ksimd::is_NaN(lhs.v[I]) && ksimd::is_NaN(rhs.v[I]))
                              ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                              : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) not_NaN(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((!ksimd::is_NaN(lhs.v[I]) && !ksimd::is_NaN(rhs.v[I]))
                              ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                              : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) any_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((ksimd::is_finite(lhs.v[I]) || ksimd::is_finite(rhs.v[I]))
                              ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                              : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_point<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Mask<Tag>) all_finite(Tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Mask<Tag>
        {
            return { ((ksimd::is_finite(lhs.v[I]) && ksimd::is_finite(rhs.v[I]))
                              ? OneBlock<same_bits_uint_t<tag_scalar_t<Tag>>>
                              : ZeroBlock<same_bits_uint_t<tag_scalar_t<Tag>>>)... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }
#pragma endregion

#pragma region--- float32 only ---
    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) rcp(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { (static_cast<tag_scalar_t<Tag>>(1.0f) / v.v[I])... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }

    template<typename Tag>
        requires (is_tag_float_32bits<Tag> && is_tag_full_and_fixed128<Tag>)
    KSIMD_API(Batch<Tag>) rsqrt(Tag, Batch<Tag> v) noexcept
    {
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<Tag>
        {
            return { (ksimd::rsqrt(v.v[I]))... };
        }(std::make_index_sequence<lanes(Tag{})>{});
    }
#pragma endregion

} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

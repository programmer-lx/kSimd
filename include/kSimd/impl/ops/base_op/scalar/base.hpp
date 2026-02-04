#pragma once

// 这个是标量的通用函数模板，如果某些类型，比如int16有专用的函数，再新建一个类，继承这个base就OK了

#include <cmath>
#include <cstring> // memcpy, memset

#include <bit> // std::bit_cast
#include <utility> // std::index_sequence

#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/base_op/BaseOp.hpp"
#include "kSimd/impl/ops/vector_types/scalar.hpp"
#include "kSimd/impl/number.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_API(...) KSIMD_OP_SCALAR_API static __VA_ARGS__ KSIMD_CALL_CONV

// -------------------------------- operators --------------------------------
namespace vector_scalar
{
    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator+(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { (lhs.v[I] + rhs.v[I])... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator-(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { (lhs.v[I] - rhs.v[I])... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator*(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { (lhs.v[I] * rhs.v[I])... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator/(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        KSIMD_WARNING_PUSH
        KSIMD_IGNORE_WARNING_MSVC(4723) // ignore n / 0 warning

        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { (lhs.v[I] / rhs.v[I])... };
        }(std::make_index_sequence<Lanes>{});

        KSIMD_WARNING_POP
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator-(Batch<S, V, A> v) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { (-v.v[I])... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator&(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) & bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator|(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) | bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator^(Batch<S, V, A> lhs, Batch<S, V, A> rhs) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) ^ bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>) operator~(Batch<S, V, A> v) noexcept
    {
        using traits = BaseOpTraits<SimdInstruction::Scalar, S>;
        constexpr auto Lanes = traits::Lanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, V, A>
        {
            return { std::bit_cast<S>(~bitcast_to_uint(v.v[I]))... };
        }(std::make_index_sequence<Lanes>{});
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator+=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator-=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator*=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator/=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator&=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator|=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    template<is_scalar_type S, size_t V, size_t A>
    KSIMD_API(Batch<S, V, A>&) operator^=(Batch<S, V, A>& lhs, Batch<S, V, A> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace vector_scalar

namespace detail
{
    /**
     * @brief 所有数据类型都有的函数
     */
    template<SimdInstruction Instruction, is_scalar_type Scalar>
    struct BaseOp_Scalar_Base
    {
        KSIMD_DETAIL_BASE_OP_TRAITS(Instruction, Scalar)

#if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(scalar_t* mem, mask_t mask) noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, mask.m, size);
        }
        KSIMD_API(mask_t) test_load_mask(const scalar_t* mem) noexcept
        {
            constexpr size_t size = Lanes * sizeof(scalar_t);
            mask_t result{};
            memcpy(result.m, mem, size);
            return result;
        }
#endif

#pragma region lane mask 通道掩码
        /**
         * @return for lane in mask, mask[0, count-1] = 1, mask[count, rest) = 0
         */
        KSIMD_API(mask_t) mask_from_lanes(size_t count) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (I < count ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }
#pragma endregion

#pragma region memory 内存操作
        /**
         * @return foreach i in lanes: mem[i] = v[i]
         * @note mem **MUST** align to sizeof(batch_t)
         */
        KSIMD_API(batch_t) load(const scalar_t* mem) noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            batch_t result{};
            memcpy(result.v, mem, size);
            return result;
        }

        /**
         * @return foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_API(batch_t) loadu(const scalar_t* mem) noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            batch_t result{};
            memcpy(result.v, mem, size);
            return result;
        }

        /**
         * @brief foreach i in lanes: mem[i] = v[i]
         * @note mem **MUST** align to sizeof(batch_t)
         */
        KSIMD_API(void) store(scalar_t* mem, batch_t v) noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, v.v, size);
        }

        /**
         * @brief foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_API(void) storeu(scalar_t* mem, batch_t v) noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, v.v, size);
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i], else: mem[i] = 0
         */
        KSIMD_API(batch_t) mask_load(const scalar_t* mem, mask_t mask) noexcept
        {
            using uint = same_bits_uint_t<scalar_t>;
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (((std::bit_cast<uint>(mask.m[I]) & OneBlock<uint>) != 0) ? mem[I] : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i], else: mem[i] = default_value
         */
        KSIMD_API(batch_t) mask_load(const scalar_t* mem, mask_t mask, batch_t default_value) noexcept
        {
            using uint = same_bits_uint_t<scalar_t>;
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (((std::bit_cast<uint>(mask.m[I]) & OneBlock<uint>) != 0) ? mem[I] : default_value.v[I])... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i], else: mem[i] = 0
         */
        KSIMD_API(batch_t) mask_loadu(const scalar_t* mem, mask_t mask) noexcept
        {
            return mask_load(mem, mask);
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i], else: mem[i] = default_value
         */
        KSIMD_API(batch_t) mask_loadu(const scalar_t* mem, mask_t mask, batch_t default_value) noexcept
        {
            return mask_load(mem, mask, default_value);
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i]
         */
        KSIMD_API(void) mask_store(scalar_t* mem, batch_t v, mask_t mask) noexcept
        {
            using uint = same_bits_uint_t<scalar_t>;

            [&]<size_t... I>(std::index_sequence<I...>)
            {
                (((std::bit_cast<uint>(mask.m[I]) & OneBlock<uint>) != 0 ? (mem[I] = v.v[I], void()) : void()), ...);
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @brief foreach i in lanes|mask: if (mask[i] == 1): mem[i] = v[i]
         */
        KSIMD_API(void) mask_storeu(scalar_t* mem, batch_t v, mask_t mask) noexcept
        {
            mask_store(mem, v, mask);
        }
#pragma endregion

#pragma region set 设置, 初始化
        /**
         * @return a memory block
         */
        KSIMD_API(batch_t) undefined() noexcept
        {
            return {};
        }

        /**
         * @return foreach i in lanes: result[i] = 0
         */
        KSIMD_API(batch_t) zero() noexcept
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            batch_t result{};
            memset(result.v, 0x00, size);
            return result;
        }

        /**
         * @return foreach i in lanes: result[i] = x
         */
        KSIMD_API(batch_t) set(scalar_t x) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { ((void)I, x)... };
            }(std::make_index_sequence<Lanes>{});
        }
#pragma endregion

#pragma region arithmetic 算术
        /**
         * @return foreach i in lanes: lhs[i] + rhs[i]
         */
        KSIMD_API(batch_t) add(batch_t lhs, batch_t rhs) noexcept
        {
            return lhs + rhs;
        }

        /**
         * @return foreach i in lanes: lhs[i] - rhs[i]
         */
        KSIMD_API(batch_t) sub(batch_t lhs, batch_t rhs) noexcept
        {
            return lhs - rhs;
        }

        /**
         * @return foreach i in lanes: lhs[i] * rhs[i]
         */
        KSIMD_API(batch_t) mul(batch_t lhs, batch_t rhs) noexcept
        {
            return lhs * rhs;
        }

        /**
         * @return lane[0] + lane[1] + ... + lane[N]
         */
        KSIMD_API(scalar_t) reduce_add(batch_t v) noexcept
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> scalar_t
            {
                return (v.v[I] + ...);
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: a[i] * b[i] + c[i]
         */
        KSIMD_API(batch_t) mul_add(batch_t a, batch_t b, batch_t c) noexcept
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (a.v[I] * b.v[I] + c.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = abx(v[i])
         */
        KSIMD_API(batch_t) abs(batch_t v) noexcept
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (std::abs(v.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = min(lhs[i], rhs[i])
         */
        KSIMD_API(batch_t) min(batch_t lhs, batch_t rhs) noexcept
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (KSIMD_NAMESPACE_NAME::min(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = max(lhs[i], rhs[i])
         */
        KSIMD_API(batch_t) max(batch_t lhs, batch_t rhs) noexcept
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (KSIMD_NAMESPACE_NAME::max(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }
#pragma endregion

#pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] == rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(mask_t) equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] == rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] != rhs[i] ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(mask_t) not_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] != rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] > rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(mask_t) greater(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] > rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] >= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(mask_t) greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] >= rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] < rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(mask_t) less(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] < rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] <= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(mask_t) less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] <= rhs.v[I] ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }
#pragma endregion

#pragma region logic 逻辑
        /**
         * @return foreach i in lanes: ~v[i]
         */
        KSIMD_API(batch_t) bit_not(batch_t v) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(v.v[0]))) == sizeof(v.v[0]),
                          "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(~KSIMD_NAMESPACE_NAME::bitcast_to_uint(v.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] & rhs[i]
         */
        KSIMD_API(batch_t) bit_and(batch_t lhs, batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) &
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: (~lhs[i]) & rhs[i]
         */
        KSIMD_API(batch_t) bit_and_not(batch_t lhs, batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(~KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) &
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] | rhs[i]
         */
        KSIMD_API(batch_t) bit_or(batch_t lhs, batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) |
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] ^ rhs[i]
         */
        KSIMD_API(batch_t) bit_xor(batch_t lhs, batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) ^
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in bits: result[i] = (mask[i] == 1) ? a[i] : b[i]
         */
        KSIMD_API(batch_t) bit_select(batch_t mask, batch_t a, batch_t b) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[0]))) == sizeof(mask.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                // 1 & any = any
                // 0 & any = 0
                // 0 | any = any

                // 假设 mask == 1，那么 mask & a = a, ~mask & b = 0
                // ret = a | 0 = a
                // 假设 mask == 0, 那么 mask & a = 0, ~mask & b = b
                // ret = 0 | b = b
                return { (std::bit_cast<scalar_t>((KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(a.v[I])) |
                                                  (~KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(b.v[I]))))... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in bits: result[i] = (mask[i] == 1) ? a[i] : b[i]
         */
        KSIMD_API(batch_t) mask_select(mask_t mask, batch_t a, batch_t b) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[0]))) == sizeof(mask.m[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (std::bit_cast<scalar_t>((KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(a.v[I])) |
                                                  (~KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(b.v[I]))))... };
            }(std::make_index_sequence<Lanes>{});
        }
#pragma endregion
    };

    /**
     * @brief 只有 float32, float64 数据类型才有的函数
     */
    template<SimdInstruction Instruction, is_scalar_floating_point FloatingPoint>
    struct BaseOp_Scalar_FloatingPoint_Base : BaseOp_Scalar_Base<Instruction, FloatingPoint>
    {
        KSIMD_DETAIL_BASE_OP_TRAITS(Instruction, FloatingPoint)

#pragma region arithmetic 算术
        /**
         * @return foreach i in lanes: lhs[i] / rhs[i]
         */
        KSIMD_API(batch_t) div(batch_t lhs, batch_t rhs) noexcept
        {
            return lhs / rhs;
        }

        /**
         * @return foreach i in lanes: 1.0 / v[i]
         */
        KSIMD_API(batch_t) one_div(batch_t v) noexcept
        {
            KSIMD_WARNING_PUSH
            KSIMD_IGNORE_WARNING_MSVC(4723) // ignore n / 0 warning

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (static_cast<scalar_t>(1) / v.v[I])... };
            }(std::make_index_sequence<Lanes>{});

            KSIMD_WARNING_POP
        }

        /**
         * @return foreach i in lanes: sqrt(v[i])
         */
        KSIMD_API(batch_t) sqrt(batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: 1.0 / sqrt(v[i])
         */
        KSIMD_API(batch_t) rsqrt(batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (static_cast<scalar_t>(1) / std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: 四舍五入
         * @note 2.5 -> 3.0; -2.5 -> -3.0
         * @warning 一般的计算使用round_nearest就够了，因为四舍五入并不是IEEE754的最近值，round需要多条指令模拟
         */
        template<RoundingMode mode>
        KSIMD_API(batch_t) round(batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
                {
                    return { (std::ceil(v.v[I]))... };
                }(std::make_index_sequence<Lanes>{});
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
                {
                    return { (std::floor(v.v[I]))... };
                }(std::make_index_sequence<Lanes>{});
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
                {
                    return { (std::nearbyint(v.v[I]))... };
                }(std::make_index_sequence<Lanes>{});
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
                {
                    return { (std::round(v.v[I]))... };
                }(std::make_index_sequence<Lanes>{});
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
                {
                    return { (std::trunc(v.v[I]))... };
                }(std::make_index_sequence<Lanes>{});
            }
        }
#pragma endregion

#pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = !(lhs[i] > rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(mask_t) not_greater(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (!(lhs.v[I] > rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] >= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(mask_t) not_greater_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (!(lhs.v[I] >= rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] < rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(mask_t) not_less(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (!(lhs.v[I] < rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] <= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(mask_t) not_less_equal(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (!(lhs.v[I] <= rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] == NaN || rhs[i] == NaN) ? 1 : 0
         */
        KSIMD_API(mask_t) any_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (is_NaN(lhs.v[I]) || is_NaN(rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] == NaN && rhs[i] == NaN) ? 1 : 0
         */
        KSIMD_API(mask_t) all_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (is_NaN(lhs.v[I]) && is_NaN(rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN && rhs[i] != NaN) ? 1 : 0
         */
        KSIMD_API(mask_t) not_NaN(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (!(is_NaN(lhs.v[I]) || is_NaN(rhs.v[I])) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN,Inf || rhs[i] != NaN,Inf) ? 1 : 0
         */
        KSIMD_API(mask_t) any_finite(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (is_finite(lhs.v[I]) || is_finite(rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN,Inf && rhs[i] != NaN,Inf) ? 1 : 0
         */
        KSIMD_API(mask_t) all_finite(batch_t lhs, batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (is_finite(lhs.v[I]) && is_finite(rhs.v[I]) ? OneBlock<scalar_t> : ZeroBlock<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }
#pragma endregion
    };
} // namespace detail

#undef KSIMD_API

KSIMD_NAMESPACE_END

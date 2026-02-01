#pragma once

// 这个是标量的通用函数模板，如果某些类型，比如int16有专用的函数，再新建一个类，继承这个base就OK了

#include <cmath>
#include <cstring> // memcpy, memset

#include <bit> // std::bit_cast
#include <utility> // std::index_sequence

#include "types.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    /**
     * @brief 所有数据类型都有的函数
     */
    template<SimdInstruction Instruction, is_scalar_type Scalar>
    struct SimdOp_Scalar_Base
    {
        KSIMD_DETAIL_SIMD_OP_TRAITS(Instruction, Scalar)

        #if defined(KSIMD_IS_TESTING)
        KSIMD_OP_SIG_SCALAR(void, test_store_mask, (scalar_t* mem, mask_t mask))
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, mask.m, size);
        }
        #endif

        #pragma region lane mask 通道掩码
        /**
         * @return for lane in mask, mask[0, count-1] = 1, mask[count, rest) = 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, mask_from_lanes, (unsigned int count))
        {
            constexpr auto lanes = traits::Lanes;

            count = count > lanes ? lanes : count;

            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (I < count ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<lanes>{});
        }
        #pragma endregion

        #pragma region memory 内存操作
        /**
         * @return foreach i in lanes: mem[i] = v[i]
         * @note mem **MUST** align to sizeof(batch_t)
         */
        KSIMD_OP_SIG_SCALAR(batch_t, load, (const scalar_t* mem))
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            batch_t result{};
            memcpy(result.v, mem, size);
            return result;
        }

        /**
         * @return foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, loadu, (const scalar_t* mem))
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
        KSIMD_OP_SIG_SCALAR(void, store, (scalar_t* mem, batch_t v))
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, v.v, size);
        }

        /**
         * @brief foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_OP_SIG_SCALAR(void, storeu, (scalar_t* mem, batch_t v))
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            memcpy(mem, v.v, size);
        }

        /**
         * @brief foreach i in lanes, j in mask: if (mask[j] != 0): mem[i] = v[i], else: mem[i] = 0
         */
        KSIMD_OP_SIG_SCALAR(batch_t, load_masked, (const scalar_t* mem, mask_t mask))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (mask.m[I] != zero_block<scalar_t> ? mem[I] : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @brief foreach i in lanes, j in mask: if (mask[j] != 0): mem[i] = v[i], else: mem[i] = default_value
         */
        KSIMD_OP_SIG_SCALAR(batch_t, load_masked, (const scalar_t* mem, mask_t mask, batch_t default_value))
        {
            return {};// TODO
        }
        #pragma endregion

        #pragma region set 设置, 初始化
        /**
         * @return foreach i in lanes: result[i] = 0
         */
        KSIMD_OP_SIG_SCALAR(batch_t, zero, ())
        {
            constexpr size_t size = traits::Lanes * sizeof(scalar_t);
            batch_t result{};
            memset(result.v, 0x00, size);
            return result;
        }

        /**
         * @return foreach i in lanes: result[i] = x
         */
        KSIMD_OP_SIG_SCALAR(batch_t, set, (scalar_t x))
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
        KSIMD_OP_SIG_SCALAR(batch_t, add, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (lhs.v[I] + rhs.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] - rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, sub, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (lhs.v[I] - rhs.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] * rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, mul, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (lhs.v[I] * rhs.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return lane[0] + lane[1] + ... + lane[N]
         */
        KSIMD_OP_SIG_SCALAR(scalar_t, reduce_sum, (batch_t v))
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
        KSIMD_OP_SIG_SCALAR(batch_t, mul_add, (batch_t a, batch_t b, batch_t c))
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
        KSIMD_OP_SIG_SCALAR(batch_t, abs, (batch_t v))
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
        KSIMD_OP_SIG_SCALAR(batch_t, min, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (detail::min(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = max(lhs[i], rhs[i])
         */
        KSIMD_OP_SIG_SCALAR(batch_t, max, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (detail::max(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }
        #pragma endregion

        #pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] == rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] == rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] != rhs[i] ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] != rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] > rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, greater, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] > rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] >= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, greater_equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] >= rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] < rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, less, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] < rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] <= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, less_equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { (lhs.v[I] <= rhs.v[I] ? one_block<scalar_t> : zero_block<scalar_t>)... };
            }(std::make_index_sequence<Lanes>{});
        }
        #pragma endregion

        #pragma region logic 逻辑
        /**
         * @return foreach i in lanes: ~v[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_not, (batch_t v))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(v.v[0]))) == sizeof(v.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(~detail::bitcast_to_uint(v.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] & rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_and, (batch_t lhs, batch_t rhs))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>(detail::bitcast_to_uint(lhs.v[I]) & detail::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: (~lhs[i]) & rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_and_not, (batch_t lhs, batch_t rhs))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>( ~detail::bitcast_to_uint(lhs.v[I]) & detail::bitcast_to_uint(rhs.v[I]) )... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] | rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_or, (batch_t lhs, batch_t rhs))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>( detail::bitcast_to_uint(lhs.v[I]) | detail::bitcast_to_uint(rhs.v[I]) )... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] ^ rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_xor, (batch_t lhs, batch_t rhs))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::bit_cast<scalar_t>( detail::bitcast_to_uint(lhs.v[I]) ^ detail::bitcast_to_uint(rhs.v[I]) )... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in bits: result[i] = (mask[i] == 1) ? a[i] : b[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, bit_select, (batch_t mask, batch_t a, batch_t b))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(mask.v[0]))) == sizeof(mask.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                // 1 & any = any
                // 0 & any = 0
                // 0 | any = any

                // 假设 mask == 1，那么 mask & a = a, ~mask & b = 0
                // ret = a | 0 = a
                // 假设 mask == 0, 那么 mask & a = 0, ~mask & b = b
                // ret = 0 | b = b
                return {
                    (
                        std::bit_cast<scalar_t>(
                            (detail::bitcast_to_uint(mask.v[I]) & detail::bitcast_to_uint(a.v[I])) |
                            (~detail::bitcast_to_uint(mask.v[I]) & detail::bitcast_to_uint(b.v[I])))
                    )...
                };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = (mask[i].sign_bit == 1) ? a[i] : b[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, sign_bit_select, (batch_t sign_mask, batch_t a, batch_t b))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(sign_mask.v[0]))) == sizeof(sign_mask.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { ( (detail::bitcast_to_uint(sign_mask.v[I]) & sign_bit_mask<detail::same_bits_uint_t<scalar_t>>) ? a.v[I] : b.v[I] )... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = (lane_mask[i] != 0) ? a[i] : b[i]
         * @note NaN != 0, so (lane_mask[i] == NaN) ? a[i] : b[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, lane_select, (batch_t lane_mask, batch_t a, batch_t b))
        {
            static_assert(sizeof(decltype(detail::bitcast_to_uint(lane_mask.v[0]))) == sizeof(lane_mask.v[0]), "byte size should be equals.");

            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { ( (lane_mask.v[I] != static_cast<scalar_t>(0)) ? a.v[I] : b.v[I] )... };
            }(std::make_index_sequence<lanes>{});
        }
        #pragma endregion
    };

    /**
     * @brief 只有 float32, float64 数据类型才有的函数
     */
    template<SimdInstruction Instruction, is_scalar_floating_point FloatingPoint>
    struct SimdOp_Scalar_FloatingPoint_Base : SimdOp_Scalar_Base<Instruction, FloatingPoint>
    {
        KSIMD_DETAIL_SIMD_OP_TRAITS(Instruction, FloatingPoint)

        #pragma region arithmetic 算术
        /**
         * @return foreach i in lanes: lhs[i] / rhs[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, div, (batch_t lhs, batch_t rhs))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (lhs.v[I] / rhs.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: 1.0 / v[i]
         */
        KSIMD_OP_SIG_SCALAR(batch_t, one_div, (batch_t v))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (static_cast<scalar_t>(1) / v.v[I])... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: sqrt(v[i])
         */
        KSIMD_OP_SIG_SCALAR(batch_t, sqrt, (batch_t v))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }

        /**
         * @return foreach i in lanes: 1.0 / sqrt(v[i])
         */
        KSIMD_OP_SIG_SCALAR(batch_t, rsqrt, (batch_t v))
        {
            constexpr auto lanes = traits::Lanes;

            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { (static_cast<scalar_t>(1) / std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<lanes>{});
        }
        #pragma endregion

        #pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = !(lhs[i] > rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_greater, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( !(lhs.v[I] > rhs.v[I]) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] >= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_greater_equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( !(lhs.v[I] >= rhs.v[I]) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] < rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_less, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( !(lhs.v[I] < rhs.v[I]) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] <= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_less_equal, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( !(lhs.v[I] <= rhs.v[I]) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] == NaN || rhs[i] == NaN) ? 1 : 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, any_NaN, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( std::isnan(lhs.v[I]) || std::isnan(rhs.v[I]) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN && rhs[i] != NaN) ? 1 : 0
         */
        KSIMD_OP_SIG_SCALAR(mask_t, not_NaN, (batch_t lhs, batch_t rhs))
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> mask_t
            {
                return { ( !(std::isnan(lhs.v[I]) || std::isnan(rhs.v[I])) ? one_block<scalar_t> : zero_block<scalar_t> )... };
            }(std::make_index_sequence<Lanes>{});
        }
        #pragma endregion
    };
}

KSIMD_NAMESPACE_END

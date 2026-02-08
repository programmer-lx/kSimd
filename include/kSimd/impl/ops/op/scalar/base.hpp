#pragma once

// 这个是标量的通用函数模板，如果某些类型，比如int16有专用的函数，再新建一个类，继承这个base就OK了

#include <cmath>
#include <cstring> // memcpy, memset

#include <bit> // std::bit_cast
#include <utility> // std::index_sequence

#include "fp16.h"

#include "traits.hpp"
#include "kSimd/impl/func_attr.hpp"
#include "kSimd/impl/ops/op/Op.hpp"
#include "kSimd/impl/ops/vector_types/scalar.hpp"
#include "kSimd/impl/number.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_API(...) KSIMD_OP_SCALAR_API static __VA_ARGS__ KSIMD_CALL_CONV

namespace detail
{
    /**
     * @brief 所有数据类型都有的函数
     */
    template<typename Traits>
    struct Executor_Scalar_Base
    {
#if defined(KSIMD_IS_TESTING)
        KSIMD_API(void) test_store_mask(typename Traits::scalar_t* mem, typename Traits::mask_t mask) noexcept
        {
            constexpr size_t size = Traits::TotalLanes * sizeof(typename Traits::scalar_t);
            memcpy(mem, mask.m, size);
        }
        KSIMD_API(typename Traits::mask_t) test_load_mask(const typename Traits::scalar_t* mem) noexcept
        {
            constexpr size_t size = Traits::TotalLanes * sizeof(typename Traits::scalar_t);
            typename Traits::mask_t result{};
            memcpy(result.m, mem, size);
            return result;
        }
#endif

#pragma region memory 内存操作
        /**
         * @return foreach i in lanes: mem[i] = v[i]
         * @note mem **MUST** align to sizeof(typename Traits::batch_t)
         */
        KSIMD_API(typename Traits::batch_t) load(const typename Traits::scalar_t* mem) noexcept
        {
            constexpr size_t size = Traits::TotalLanes * sizeof(typename Traits::scalar_t);
            typename Traits::batch_t result{};
            memcpy(result.v, mem, size);
            return result;
        }

        /**
         * @return foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_API(typename Traits::batch_t) loadu(const typename Traits::scalar_t* mem) noexcept
        {
            return load(mem);
        }

        /**
         * @return load [mem : mem + count * sizeof(scalar_t)]
         */
        KSIMD_API(typename Traits::batch_t) load_partial(const typename Traits::scalar_t* mem, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;

            const size_t size = count * sizeof(typename Traits::scalar_t);

            if (size == 0)
                return zero();

            typename Traits::batch_t result{};
            memcpy(result.v, mem, size);
            return result;
        }

        /**
         * @brief foreach i in lanes: mem[i] = v[i]
         * @note mem **MUST** align to sizeof(typename Traits::batch_t)
         */
        KSIMD_API(void) store(typename Traits::scalar_t* mem, typename Traits::batch_t v) noexcept
        {
            constexpr size_t size = Traits::TotalLanes * sizeof(typename Traits::scalar_t);
            memcpy(mem, v.v, size);
        }

        /**
         * @brief foreach i in lanes: mem[i] = v[i]
         */
        KSIMD_API(void) storeu(typename Traits::scalar_t* mem, typename Traits::batch_t v) noexcept
        {
            store(mem, v);
        }

        KSIMD_API(void) store_partial(typename Traits::scalar_t* mem, typename Traits::batch_t v, size_t count) noexcept
        {
            count = count > Traits::TotalLanes ? Traits::TotalLanes : count;

            if (count == 0)
                return;

            const size_t size = count * sizeof(typename Traits::scalar_t);
            memcpy(mem, v.v, size);
        }
#pragma endregion

#pragma region set 设置, 初始化
        /**
         * @return a memory block
         */
        KSIMD_API(typename Traits::batch_t) undefined() noexcept
        {
            return {};
        }

        /**
         * @return foreach i in lanes: result[i] = 0
         */
        KSIMD_API(typename Traits::batch_t) zero() noexcept
        {
            constexpr size_t size = Traits::TotalLanes * sizeof(typename Traits::scalar_t);
            typename Traits::batch_t result{};
            memset(result.v, 0x00, size);
            return result;
        }

        /**
         * @return foreach i in lanes: result[i] = x
         */
        KSIMD_API(typename Traits::batch_t) set(typename Traits::scalar_t x) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { ((void)I, x)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
#pragma endregion

#pragma region arithmetic 算术
        /**
         * @return foreach i in lanes: lhs[i] + rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) add(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return lhs + rhs;
        }

        /**
         * @return foreach i in lanes: lhs[i] - rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) sub(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return lhs - rhs;
        }

        /**
         * @return foreach i in lanes: lhs[i] * rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) mul(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return lhs * rhs;
        }

        /**
         * @return foreach i in lanes: a[i] * b[i] + c[i]
         */
        KSIMD_API(typename Traits::batch_t) mul_add(typename Traits::batch_t a, typename Traits::batch_t b, typename Traits::batch_t c) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (a.v[I] * b.v[I] + c.v[I])... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = min(lhs[i], rhs[i])
         */
        KSIMD_API(typename Traits::batch_t) min(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (KSIMD_NAMESPACE_NAME::min(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = max(lhs[i], rhs[i])
         */
        KSIMD_API(typename Traits::batch_t) max(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (KSIMD_NAMESPACE_NAME::max(lhs.v[I], rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
#pragma endregion

#pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] == rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(typename Traits::mask_t) equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] == rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] != rhs[i] ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(typename Traits::mask_t) not_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] != rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] > rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(typename Traits::mask_t) greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] > rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] >= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(typename Traits::mask_t) greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] >= rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes, j in mask: result[j] = lhs[i] < rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(typename Traits::mask_t) less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] < rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] <= rhs[i] ? 1 : 0
         * @note if NaN: return 0
         */
        KSIMD_API(typename Traits::mask_t) less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (lhs.v[I] <= rhs.v[I] ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
#pragma endregion

#pragma region logic 逻辑
        /**
         * @return foreach i in lanes: ~v[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_not(typename Traits::batch_t v) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(v.v[0]))) == sizeof(v.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { std::bit_cast<typename Traits::scalar_t>(~KSIMD_NAMESPACE_NAME::bitcast_to_uint(v.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] & rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_and(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { std::bit_cast<typename Traits::scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) &
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (~lhs[i]) & rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_and_not(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { std::bit_cast<typename Traits::scalar_t>(~KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) &
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] | rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_or(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { std::bit_cast<typename Traits::scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) |
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: lhs[i] ^ rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_xor(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[0]))) == sizeof(lhs.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { std::bit_cast<typename Traits::scalar_t>(KSIMD_NAMESPACE_NAME::bitcast_to_uint(lhs.v[I]) ^
                                                 KSIMD_NAMESPACE_NAME::bitcast_to_uint(rhs.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in bits: result[i] = (mask[i] == 1) ? a[i] : b[i]
         */
        KSIMD_API(typename Traits::batch_t) bit_select(typename Traits::batch_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[0]))) == sizeof(mask.v[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                // 1 & any = any
                // 0 & any = 0
                // 0 | any = any

                // 假设 mask == 1，那么 mask & a = a, ~mask & b = 0
                // ret = a | 0 = a
                // 假设 mask == 0, 那么 mask & a = 0, ~mask & b = b
                // ret = 0 | b = b
                return { (std::bit_cast<typename Traits::scalar_t>((KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(a.v[I])) |
                                                  (~KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.v[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(b.v[I]))))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in bits: result[i] = (mask[i] == 1) ? a[i] : b[i]
         */
        KSIMD_API(typename Traits::batch_t) mask_select(typename Traits::mask_t mask, typename Traits::batch_t a, typename Traits::batch_t b) noexcept
        {
            static_assert(sizeof(decltype(KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[0]))) == sizeof(mask.m[0]),
                          "byte size should be equals.");

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (std::bit_cast<typename Traits::scalar_t>((KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(a.v[I])) |
                                                  (~KSIMD_NAMESPACE_NAME::bitcast_to_uint(mask.m[I]) &
                                                   KSIMD_NAMESPACE_NAME::bitcast_to_uint(b.v[I]))))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
#pragma endregion
    };

    /**
     * @brief 只有有符号的类型才有的函数
     */
    template<typename Traits>
    struct Executor_Scalar_Signed_Base : Executor_Scalar_Base<Traits>
    {
        /**
         * @return foreach i in lanes: result[i] = |v[i]|
         */
        KSIMD_API(typename Traits::batch_t) abs(typename Traits::batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (std::abs(v.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: result[i] = -v[i]
         */
        KSIMD_API(typename Traits::batch_t) neg(typename Traits::batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (-v.v[I])... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
    };

    /**
     * @brief 只有 float16, float32, float64 数据类型才有的函数
     */
    template<typename Traits>
    struct Executor_Scalar_FloatingPoint_Base
        : Executor_Scalar_Signed_Base<Traits>
        , OpHelper
    {
#pragma region arithmetic 算术
        /**
         * @return foreach i in lanes: lhs[i] / rhs[i]
         */
        KSIMD_API(typename Traits::batch_t) div(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return lhs / rhs;
        }

        /**
         * @return foreach i in lanes: 1.0 / v[i]
         */
        KSIMD_API(typename Traits::batch_t) one_div(typename Traits::batch_t v) noexcept
        {
            KSIMD_WARNING_PUSH
            KSIMD_IGNORE_WARNING_MSVC(4723) // ignore n / 0 warning

            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (static_cast<typename Traits::scalar_t>(1) / v.v[I])... };
            }(std::make_index_sequence<Traits::TotalLanes>{});

            KSIMD_WARNING_POP
        }

        /**
         * @return foreach i in lanes: sqrt(v[i])
         */
        KSIMD_API(typename Traits::batch_t) sqrt(typename Traits::batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: 1.0 / sqrt(v[i])
         */
        KSIMD_API(typename Traits::batch_t) rsqrt(typename Traits::batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (static_cast<typename Traits::scalar_t>(1) / std::sqrt(v.v[I]))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @brief
         * RoundingMode::Up: 向上取整 \n
         * RoundingMode::Down: 向下取整 \n
         * RoundingMode::Nearest: 最近偶数 \n
         * RoundingMode::Round: 四舍五入 \n
         * RoundingMode::ToZero: 向0取整 \n
         */
        template<RoundingMode mode>
        KSIMD_API(typename Traits::batch_t) round(typename Traits::batch_t v) noexcept
        {
            if constexpr (mode == RoundingMode::Up)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
                {
                    return { (std::ceil(v.v[I]))... };
                }(std::make_index_sequence<Traits::TotalLanes>{});
            }
            else if constexpr (mode == RoundingMode::Down)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
                {
                    return { (std::floor(v.v[I]))... };
                }(std::make_index_sequence<Traits::TotalLanes>{});
            }
            else if constexpr (mode == RoundingMode::Nearest)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
                {
                    return { (std::nearbyint(v.v[I]))... };
                }(std::make_index_sequence<Traits::TotalLanes>{});
            }
            else if constexpr (mode == RoundingMode::Round)
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
                {
                    return { (std::round(v.v[I]))... };
                }(std::make_index_sequence<Traits::TotalLanes>{});
            }
            else /* if constexpr (mode == RoundingMode::ToZero) */
            {
                return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
                {
                    return { (std::trunc(v.v[I]))... };
                }(std::make_index_sequence<Traits::TotalLanes>{});
            }
        }
#pragma endregion

#pragma region compare 比较
        /**
         * @return foreach i in lanes, j in mask: result[j] = !(lhs[i] > rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(typename Traits::mask_t) not_greater(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (!(lhs.v[I] > rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] >= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(typename Traits::mask_t) not_greater_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (!(lhs.v[I] >= rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] < rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(typename Traits::mask_t) not_less(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (!(lhs.v[I] < rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: !(lhs[i] <= rhs[i]) ? 1 : 0
         * @note if NaN: return 1
         */
        KSIMD_API(typename Traits::mask_t) not_less_equal(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (!(lhs.v[I] <= rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] == NaN || rhs[i] == NaN) ? 1 : 0
         */
        KSIMD_API(typename Traits::mask_t) any_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (is_NaN(lhs.v[I]) || is_NaN(rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] == NaN && rhs[i] == NaN) ? 1 : 0
         */
        KSIMD_API(typename Traits::mask_t) all_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (is_NaN(lhs.v[I]) && is_NaN(rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN && rhs[i] != NaN) ? 1 : 0
         */
        KSIMD_API(typename Traits::mask_t) not_NaN(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (!(is_NaN(lhs.v[I]) || is_NaN(rhs.v[I])) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN,Inf || rhs[i] != NaN,Inf) ? 1 : 0
         */
        KSIMD_API(typename Traits::mask_t) any_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (is_finite(lhs.v[I]) || is_finite(rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return foreach i in lanes: (lhs[i] != NaN,Inf && rhs[i] != NaN,Inf) ? 1 : 0
         */
        KSIMD_API(typename Traits::mask_t) all_finite(typename Traits::batch_t lhs, typename Traits::batch_t rhs) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::mask_t
            {
                return { (is_finite(lhs.v[I]) && is_finite(rhs.v[I]) ? OneBlock<typename Traits::scalar_t> : ZeroBlock<typename Traits::scalar_t>)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
#pragma endregion
    };

    template<typename Traits>
    using Executor_Scalar_float32 = Executor_Scalar_FloatingPoint_Base<Traits>;

    template<typename Traits>
    using Executor_Scalar_float64 = Executor_Scalar_FloatingPoint_Base<Traits>;
} // namespace detail


// mixin functions
namespace detail
{
    template<typename Traits>
    struct Base_Mixin_Scalar
    {
        /**
         * @return lane[0] + lane[1] + ... + lane[N]
         */
        KSIMD_API(typename Traits::scalar_t) reduce_add(typename Traits::batch_t v) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::scalar_t
            {
                return (v.v[I] + ...);
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return [ 0, 1, 2, ... , TotalLanes - 1 ]
         */
        KSIMD_API(typename Traits::batch_t) sequence() noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { static_cast<typename Traits::scalar_t>(I)... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return [ base + 0, base + 1, base + 2, ... , base + TotalLanes - 1 ]
         */
        KSIMD_API(typename Traits::batch_t) sequence(auto base) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (base + static_cast<typename Traits::scalar_t>(I))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }

        /**
         * @return [ base + (0 * stride), base + (1 * stride), ... , base + ((TotalLanes - 1) * stride) ]
         */
        KSIMD_API(typename Traits::batch_t) sequence(auto base, auto stride) noexcept
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> typename Traits::batch_t
            {
                return { (base + (static_cast<typename Traits::scalar_t>(I) * stride))... };
            }(std::make_index_sequence<Traits::TotalLanes>{});
        }
    };
    
    template<typename Traits>
    struct Base_Mixin_Scalar_f16c
    {
        static_assert(std::is_same_v<typename Traits::scalar_t, float32>);

        /**
        * @return 加载 [mem : mem + sizeof(float16) * TotalLanes]，然后进行类型转换，将FP16提升为FP32
        */
        KSIMD_API(typename Traits::batch_t) load_float16(const float16* mem) noexcept
        {
            typename Traits::batch_t res;
            for (size_t i = 0; i < Traits::TotalLanes; ++i)
            {
                res.v[i] = fp16_ieee_to_fp32_value(std::bit_cast<uint16_t>(mem[i]));
            }
            return res;
        }

        KSIMD_API(typename Traits::batch_t) loadu_float16(const float16* mem) noexcept
        {
            return load_float16(mem);
        }

        KSIMD_API(void) store_float16(float16* mem, typename Traits::batch_t v) noexcept
        {
            for (size_t i = 0; i < Traits::TotalLanes; ++i)
            {
                mem[i] = std::bit_cast<float16>(fp16_ieee_from_fp32_value(v.v[i]));
            }
        }

        KSIMD_API(void) storeu_float16(float16* mem, typename Traits::batch_t v) noexcept
        {
            store_float16(mem, v);
        }
    };
}

// -------------------------------- operators --------------------------------
namespace vector_scalar
{
    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator+(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { (lhs.v[I] + rhs.v[I])... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator-(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { (lhs.v[I] - rhs.v[I])... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator*(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { (lhs.v[I] * rhs.v[I])... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator/(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        KSIMD_WARNING_PUSH
        KSIMD_IGNORE_WARNING_MSVC(4723) // ignore n / 0 warning

        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;

        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { (lhs.v[I] / rhs.v[I])... };
        }(std::make_index_sequence<TotalLanes>{});

        KSIMD_WARNING_POP
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator-(Batch<S, reg_count> v) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { (-v.v[I])... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator&(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) & bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator|(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) | bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator^(Batch<S, reg_count> lhs, Batch<S, reg_count> rhs) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { std::bit_cast<S>(bitcast_to_uint(lhs.v[I]) ^ bitcast_to_uint(rhs.v[I]))... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>) operator~(Batch<S, reg_count> v) noexcept
    {
        using traits = BaseOpTraits_Scalar<S, reg_count>;
        constexpr auto TotalLanes = traits::TotalLanes;
        return [&]<size_t... I>(std::index_sequence<I...>) -> Batch<S, reg_count>
        {
            return { std::bit_cast<S>(~bitcast_to_uint(v.v[I]))... };
        }(std::make_index_sequence<TotalLanes>{});
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator+=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs + rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator-=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs - rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator*=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs * rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator/=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs / rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator&=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs & rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator|=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs | rhs;
    }

    template<is_scalar_type S, size_t reg_count>
    KSIMD_API(Batch<S, reg_count>&) operator^=(Batch<S, reg_count>& lhs, Batch<S, reg_count> rhs) noexcept
    {
        return lhs = lhs ^ rhs;
    }
} // namespace vector_scalar

#undef KSIMD_API

KSIMD_NAMESPACE_END

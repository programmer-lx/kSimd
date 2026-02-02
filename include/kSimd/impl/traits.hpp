#pragma once

#include <cstdint>

#include <limits>
#include <type_traits>

#include "platform.hpp"

KSIMD_NAMESPACE_BEGIN

// 这个枚举用于SimdOp的模板参数
enum class SimdInstruction : int
{
    Scalar,

    SSE_Start,
    SSE,
    SSE2,
    SSE3,
    SSE4_1,
    SSE_End,

    AVX_Start,
    AVX,
    AVX2,
    AVX2_FMA3_F16C,
    AVX_End
};


// clang-format off

// ----------------- scalar type -----------------

// (u)int(n)
using int8   = int8_t   ;
using uint8  = uint8_t  ;
using int16  = int16_t  ;
using uint16 = uint16_t ;
using int32  = int32_t  ;
using uint32 = uint32_t ;
using int64  = int64_t  ;
using uint64 = uint64_t ;

// floating point
using float32 = float;
using float64 = double;
static_assert(sizeof(float32) == 4 && std::numeric_limits<float32>::is_iec559);
static_assert(sizeof(float64) == 8 && std::numeric_limits<float64>::is_iec559);

// clang-format on

template<typename T>
concept is_scalar_floating_point = std::is_same_v<T, float32> || std::is_same_v<T, float64>;

template<typename T>
concept is_scalar_type =
    is_scalar_floating_point<T> ||
    std::is_same_v<T, int8>     ||
    std::is_same_v<T, uint8>    ||
    std::is_same_v<T, int16>    ||
    std::is_same_v<T, uint16>   ||
    std::is_same_v<T, int32>    ||
    std::is_same_v<T, uint32>   ||
    std::is_same_v<T, int64>    ||
    std::is_same_v<T, uint64>;

template<typename T, typename... Ts>
concept is_scalar_type_includes = is_scalar_type<T> && (std::is_same_v<T, Ts> || ...);

// ----------------- batch type -----------------
namespace detail
{
    enum class UnderlyingSimdType
    {
        // Scalar
        ScalarArray, // SSE只支持float32的SIMD指令和数据类型，所以标量数组是SSE的专属

        // SSE
        m128,
        m128d,
        m128i,

        // AVX
        m256,
        m256d,
        m256i
    };
}

template<typename T>
concept is_batch_type = requires(T v)
{
    typename T::scalar_t;

    T::underlying_simd_type;
    requires std::is_same_v<std::remove_cvref_t<decltype(T::underlying_simd_type)>, detail::UnderlyingSimdType>;

    v.v;
};

template<typename T, typename... Ts>
concept is_batch_type_includes = is_batch_type<T> && (std::is_same_v<typename T::scalar_t, Ts> || ...);


// ----------------- mask type -----------------
// mask用于控制哪个lane能够被操作，比如 mask = 0b0011'1111，就代表[5:0]lanes能被操作，其他lanes就使用默认值进行填充
// 所以mask类型只能通过sizeof(scalar_t)来区分，不能通过scalar_t区分
namespace detail
{
    enum class UnderlyingMaskType
    {
        // Scalar
        ScalarArray,

        // SSE
        m128,
        m128d,
        m128i,

        // AVX
        m256,
        m256d,
        m256i,

        // after AVX-512
        mmask8,
        mmask16,
        mmask32,
        mmask64
    };
}

template<typename T>
concept is_mask_type = requires(T v)
{
    typename T::scalar_t;

    T::underlying_mask_type;
    requires std::is_same_v<std::remove_cvref_t<decltype(T::underlying_mask_type)>, detail::UnderlyingMaskType>;

    v.m;
};

template<SimdInstruction Instruction, is_scalar_type ScalarType>
struct SimdTraits;

namespace detail
{
    template<SimdInstruction Instruction, is_scalar_type S, typename BatchType, typename MaskType, size_t Alignment>
    struct SimdTraits_Base
    {
        using batch_t = BatchType;
        using scalar_t = S;
        using mask_t = MaskType;
        static constexpr SimdInstruction CurrentInstruction = Instruction;
        static constexpr size_t BatchSize = batch_t::byte_size;
        static constexpr size_t ElementSize = sizeof(scalar_t);
        static constexpr size_t Lanes = (BatchSize / ElementSize);
        static constexpr size_t BatchAlignment = Alignment;

        static_assert(BatchSize % ElementSize == 0); // 必须能整除
    };
}

#define KSIMD_DETAIL_SIMD_OP_TRAITS(instruction, scalar_type) \
    using traits = SimdTraits<instruction, scalar_type>; \
    using batch_t = typename traits::batch_t; \
    using scalar_t = typename traits::scalar_t; \
    using mask_t = typename traits::mask_t; \
    static constexpr SimdInstruction CurrentInstruction = traits::CurrentInstruction; \
    static constexpr size_t BatchSize = traits::BatchSize; \
    static constexpr size_t ElementSize = traits::ElementSize; \
    static constexpr size_t Lanes = traits::Lanes; \
    static constexpr size_t BatchAlignment = traits::BatchAlignment;

KSIMD_NAMESPACE_END

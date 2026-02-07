#pragma once

// 该文件编写所有的SIMD类型相关的逻辑

#include <cstdint>

#include <limits>
#include <type_traits>

#include "platform.hpp"
#include "dyn_instruction_name.hpp"

KSIMD_NAMESPACE_BEGIN

// 这个枚举用于SimdOp的模板参数
enum class SimdInstruction : int
{
    KSIMD_DYN_INSTRUCTION_SCALAR,

    AVX_Start,
    KSIMD_DYN_INSTRUCTION_AVX2_FMA3_F16C,
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
using float16 = uint16; // 不定义 enum class，只定义宽度
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

template<typename T>
concept is_scalar_signed = is_scalar_type<T> && std::is_signed_v<T>;

template<typename T, typename... Ts>
concept is_scalar_signed_includes = is_scalar_signed<T> && (std::is_same_v<T, Ts> || ...);

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

template<typename T, typename... Ts>
concept is_mask_type_includes = is_mask_type<T> && (std::is_same_v<typename T::scalar_t, Ts> || ...);

namespace detail
{
    template<SimdInstruction I, is_scalar_type S>
    consteval size_t reg_lanes()
    {
        constexpr size_t lanes = []() -> size_t
        {
            if constexpr (I == SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR)
            {
                return 16 / sizeof(S);
            }
            if constexpr (I > SimdInstruction::AVX_Start && I < SimdInstruction::AVX_End)
            {
                return 32 / sizeof(S);
            }
            return SIZE_MAX;
        }();
        static_assert(lanes != SIZE_MAX);

        return lanes;
    }
}

// 每个寄存器能够装下的标量的数量
template<SimdInstruction I, is_scalar_type S>
KSIMD_HEADER_GLOBAL_CONSTEXPR size_t RegLanes = detail::reg_lanes<I, S>();

namespace detail
{
    template<SimdInstruction Instruction, typename BatchType, typename MaskType, size_t Alignment>
    struct SimdTraits_Base
    {
        using batch_t = BatchType;
        using scalar_t = typename BatchType::scalar_t;
        using mask_t = MaskType;
        static constexpr SimdInstruction internal_instruction_ = Instruction; // 当前所分发的指令集，不是准确值，一般是最低值
        static constexpr size_t BatchAlignment = Alignment;                 // 对齐
        static constexpr size_t BatchBytes = batch_t::byte_size;            // 向量的字节长度
        static constexpr size_t ElementBytes = sizeof(scalar_t);            // 每个元素的字节长度
        static constexpr size_t TotalLanes = BatchBytes / ElementBytes;     // 总通道数
        static constexpr size_t RegCount = batch_t::reg_count;              // 寄存器的数量，标量特殊处理，视一个寄存器为128bit，跟SSE对齐
        static constexpr size_t RegBytes = BatchBytes / RegCount;           // 每个寄存器所占用的字节数
        static constexpr size_t RegLanes = RegBytes / ElementBytes;         // 每个寄存器能够装下的标量的数量(可用于index展开时的步长计算)

        static_assert(
            BatchBytes % ElementBytes == 0 &&
            BatchBytes % RegBytes == 0 &&
            RegBytes % RegLanes == 0
        );
    };
}

KSIMD_NAMESPACE_END

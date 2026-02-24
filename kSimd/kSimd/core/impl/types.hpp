#pragma once

#include <cstddef>
#include <cstdint>

#include <limits>
#include <type_traits>

#include "base.hpp"

#if KSIMD_SUPPORT_STD_FLOAT16 || KSIMD_SUPPORT_STD_BFLOAT16 || KSIMD_SUPPORT_STD_FLOAT32 || KSIMD_SUPPORT_STD_FLOAT64
    #include <stdfloat>
#endif

// clang-format off

namespace ksimd
{
    // ----------------- scalar type -----------------

    // floating point
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

    template<typename T>
    concept is_scalar_floating_point =
           std::is_same_v<T, float>
        || std::is_same_v<T, double>
#if KSIMD_SUPPORT_STD_FLOAT32
        || std::is_same_v<T, std::float32_t>
#endif
#if KSIMD_SUPPORT_STD_FLOAT64
        || std::is_same_v<T, std::float64_t>
#endif
        ;

    template<typename T>
    concept is_scalar_type =
        is_scalar_floating_point<T>   ||
        std::is_same_v<T, int8_t>     ||
        std::is_same_v<T, uint8_t>    ||
        std::is_same_v<T, int16_t>    ||
        std::is_same_v<T, uint16_t>   ||
        std::is_same_v<T, int32_t>    ||
        std::is_same_v<T, uint32_t>   ||
        std::is_same_v<T, int64_t>    ||
        std::is_same_v<T, uint64_t>;

    template<typename T, typename... Ts>
    concept is_scalar_type_includes = is_scalar_type<T> && (std::is_same_v<T, Ts> || ...);

    // signed types
    template<typename T>
    concept is_scalar_signed = is_scalar_type<T> && std::is_signed_v<T>;

    template<typename T, typename... Ts>
    concept is_scalar_signed_includes = is_scalar_signed<T> && (std::is_same_v<T, Ts> || ...);

    // float32
    template<typename T>
    concept is_scalar_type_float_32bits = is_scalar_type_includes<
        T
        , float
    #if KSIMD_SUPPORT_STD_FLOAT32
        , std::float32_t
    #endif
    >;

    // float64
    template<typename T>
    concept is_scalar_type_float_64bits = is_scalar_type_includes<
        T
        , double
    #if KSIMD_SUPPORT_STD_FLOAT64
        , std::float64_t
    #endif
    >;

    namespace alignment
    {
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Scalar128 = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec128 = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec256 = 32;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec512 = 64;

        // x86平台使用AVX512对齐
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Max_X86 = Vec512;

        // ARM平台: SVE要求16B对齐即可，但是缓存行可能是64B，所以提供了最低对齐和性能对齐两个选项
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Required_ARM = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Performance_ARM = 64;

        // 跨平台
    #if KSIMD_ARCH_X86_ANY
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Required = Max_X86;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Performance = Max_X86;
    #elif KSIMD_ARCH_ARM_ANY
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Required = Required_ARM;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Performance = Performance_ARM;
    #else
        #error unknown arch
    #endif
    }

    namespace vec_size
    {
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Invalid = 0;

        // 模拟Vec128
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Scalar128 = 16;

        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec128 = 16;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec256 = 32;
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Vec512 = 64;

        // 变长向量类型，使用size_t最大值表示
        KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Scalable = SIZE_MAX;
    }
}

// clang-format on

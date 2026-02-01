#pragma once

#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#include <type_traits>
#include <stdexcept>
#include <exception>
#include <iostream>
#include <string>
#include <random>
#include <chrono>

#include <gtest/gtest.h>

#if !defined(KSIMD_IS_TESTING)
#error "please define KSIMD_IS_TESTING macro to enable testing."
#endif

template<std::floating_point F>
F random_f(F min, F max)
{
    static std::mt19937_64 rng{ static_cast<unsigned long>(std::time(nullptr)) };
    std::uniform_real_distribution<F> dist(min, max);
    return dist(rng);
}

class ScopeTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    explicit ScopeTimer(std::string name = "ScopeTimer")
        : m_name(std::move(name)), m_start(Clock::now())
    {
    }

    // 析构函数：对象生命周期结束时自动触发
    ~ScopeTimer()
    {
        double elapsed = time_millis();
        std::cout << "[" << m_name << "] Elapsed time: "
                  << elapsed << " ms" << std::endl;
    }

    double time_millis() const
    {
        auto end = Clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - m_start;
        return elapsed.count();
    }

private:
    std::string m_name;
    Clock::time_point m_start;
};

bool simd_type_bit_equal(const auto& a, const auto& b)
{
    using Ta = std::remove_cvref_t<decltype(a)>;
    using Tb = std::remove_cvref_t<decltype(b)>;

    static_assert(sizeof(Ta) == sizeof(Tb));

    return (std::memcmp(&a, &b, sizeof(Ta)) == 0);
}

#define FILL_ARRAY(arr, val) \
    do { \
        for (size_t i___ = 0; i___ < std::size(arr); ++i___) { \
            (arr)[i___] = val; \
        } \
    } while (0)

template<typename T, typename T2>
bool array_bit_equal(T* arr, size_t len, const T2& val)
{
    for (size_t i = 0; i < len; ++i)
    {
        if (!simd_type_bit_equal(arr[i], val))
        {
            return false;
        }
    }

    return true;
}

template<typename T, typename T2>
bool array_equal(T* arr, size_t len, const T2& val)
{
    for (size_t i = 0; i < len; ++i)
    {
        if (arr[i] != val)
        {
            return false;
        }
    }
    return true;
}

template<typename T, typename T2, typename T3>
bool array_approximately(T* arr, size_t len, const T2& val, const T3& tolerance)
{
    for (size_t i = 0; i < len; ++i)
    {
        if (std::abs(arr[i] - val) > tolerance)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
struct float_bits;

template<>
struct float_bits<float>
{
    using uint = uint32_t;
};

template<>
struct float_bits<double>
{
    using uint = uint64_t;
};

template<typename T>
using float_bits_t = typename float_bits<T>::uint;

template<typename F>
constexpr F make_float_from_bits(float_bits_t<F> bits) noexcept
{
    return std::bit_cast<F>(bits);
}

template<size_t Bytes>
struct uint_from_bytes
{
    using type = std::conditional_t<
        (Bytes == sizeof(uint8_t)), uint8_t,
        std::conditional_t<
            (Bytes == sizeof(uint16_t)), uint16_t,
            std::conditional_t<
                (Bytes == sizeof(uint32_t)), uint32_t,
                std::conditional_t<
                    (Bytes == sizeof(uint64_t)), uint64_t, void
                >
            >
        >
    >;

    // check
    static_assert(!std::is_void_v<type>);
};

template<size_t Bytes>
using uint_from_bytes_t = typename uint_from_bytes<Bytes>::type;

template<typename S>
using same_bits_uint_t = uint_from_bytes_t<sizeof(S)>;

template<typename S>
bool test_bit(S bits, int index)
{
    using UInt = same_bits_uint_t<S>;
    auto uint = std::bit_cast<UInt>(bits);
    return ( uint & (static_cast<UInt>(1) << static_cast<UInt>(index)) ) != 0;
}

template<typename F>
F qNaN = std::numeric_limits<F>::quiet_NaN();

#define TEST_ONCE_DYN(func_name) \
    KSIMD_DYN_DISPATCH_FUNC(func_name); \
    TEST(dyn_dispatch, func_name) \
    { \
        for (size_t idx___ = 0; idx___ < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)); ++idx___) \
        { \
            KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)[idx___](); \
        } \
    }

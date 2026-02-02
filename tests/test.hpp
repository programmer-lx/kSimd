#pragma once

#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#include <bitset>
#include <type_traits>
#include <stdexcept>
#include <exception>
#include <iostream>
#include <string>
#include <random>
#include <chrono>

#include <gtest/gtest.h>

#include "kSimd/impl/utils.hpp"

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

bool bit_equal(const auto& a, const auto& b)
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
        if (!bit_equal(arr[i], val))
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
constexpr T make_var_from_bits(ksimd::same_bits_uint_t<T> bits) noexcept
{
    return std::bit_cast<T>(bits);
}

template<typename S>
bool test_bit(S bits, int index)
{
    using UInt = ksimd::same_bits_uint_t<S>;
    auto uint = std::bit_cast<UInt>(bits);
    return ( uint & (static_cast<UInt>(1) << static_cast<UInt>(index)) ) != 0;
}

template<typename F>
F qNaN = std::numeric_limits<F>::quiet_NaN();

template<typename F>
F inf = std::numeric_limits<F>::infinity();

#define TEST_ONCE_DYN(func_name) \
    KSIMD_DYN_DISPATCH_FUNC(func_name); \
    TEST(dyn_dispatch, func_name) \
    { \
        for (size_t idx___ = 0; idx___ < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)); ++idx___) \
        { \
            KSIMD_DETAIL_PFN_TABLE_FULL_NAME(func_name)[idx___](); \
        } \
    }

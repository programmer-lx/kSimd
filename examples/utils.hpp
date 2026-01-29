#pragma once

#include <chrono>
#include <iostream>
#include <random>

class Timer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    explicit Timer(const std::string& name)
        : m_name(name), m_start(Clock::now()) {}

    double get_time() const
    {
        auto end = Clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - m_start;
        return elapsed.count();
    }

private:
    std::string m_name;
    Clock::time_point m_start;
};

template<std::floating_point F>
F random_f(F min, F max)
{
    static std::mt19937_64 rng{ std::random_device{}() };
    std::uniform_real_distribution<F> dist(min, max);
    return dist(rng);
}

inline void waste_time(double microseconds)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto target = std::chrono::duration<double, std::micro>(microseconds);

    // 强制 CPU 不断轮询时钟
    while (std::chrono::high_resolution_clock::now() - start < target)
    {
        _mm_pause();
    }
}
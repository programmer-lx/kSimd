#pragma once

#include <chrono>
#include <iostream>
#include <random>

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

template<std::floating_point F>
F random_f(F min, F max)
{
    static std::mt19937_64 rng{ std::random_device{}() };
    std::uniform_real_distribution<F> dist(min, max);
    return dist(rng);
}
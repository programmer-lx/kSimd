#include <xmmintrin.h>
#include <emmintrin.h>

#include <type_traits>
#include <iostream>
#include <cstdint>

#ifdef _MSC_VER
    #define SIMD_API __forceinline
    #define NOINLINE __declspec(noinline)
    #define SSE2_TARGET
#else
    #define SIMD_API inline __attribute__((always_inline, flatten))
    #define NOINLINE __attribute__((noinline))
    #define SSE2_TARGET __attribute__((target("sse2")))
#endif

struct float32x4
{
    __m128 v;
};

struct float64x2
{
    __m128d v;
};

SIMD_API SSE2_TARGET
float32x4 loadu_f32x4(const float* mem)
{
    return {_mm_loadu_ps(mem)};
}

SIMD_API SSE2_TARGET
void storeu(double* mem, float64x2 f64x2)
{
    _mm_storeu_pd(mem, f64x2.v);
}

SIMD_API SSE2_TARGET
void storeu(float* mem, float32x4 f32x4)
{
    _mm_storeu_ps(mem, f32x4.v);
}

template<typename To, typename From>
SIMD_API SSE2_TARGET
To bitcast(From from) noexcept
{
    // f32x4 -> f64x2
    if constexpr (std::is_same_v<To, float64x2> && std::is_same_v<From, float32x4>)
    {
        return {_mm_castps_pd(from.v)};
    }
    // f64x2 -> f32x4
    else if constexpr (std::is_same_v<To, float32x4> && std::is_same_v<From, float64x2>)
    {
        return {_mm_castpd_ps(from.v)};
    }
    // self -> self
    else if constexpr (std::is_same_v<To, From>)
    {
        return from;
    }

    return To{};
}

NOINLINE
void ksimd_test_no_inline(void* mem)
{
    std::cout << reinterpret_cast<uintptr_t>(mem);
}


void test()
{
    float data[4] = { 1, 2, 3, 4 };
    float32x4 f32 = loadu_f32x4(data);
    float64x2 f64 = bitcast<float64x2>(f32);

    double result[2] = {};
    storeu(result, f64);

    ksimd_test_no_inline(result);
}

void test_self_to_self()
{
    float data[4] = { 1, 2, 3, 4 };
    float32x4 f32 = loadu_f32x4(data);
    float32x4 f32_1 = bitcast<float32x4>(f32);

    float result[4] = {};
    storeu(result, f32_1);

    ksimd_test_no_inline(result);
}
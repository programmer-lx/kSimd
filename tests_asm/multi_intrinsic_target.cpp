#include <immintrin.h>

#define FORCE_INLINE inline __attribute__((always_inline, flatten))
#define AVX_ATTR __attribute__((target("avx,fma")))
#define SSE2_ATTR __attribute__((target("sse2")))
#define NO_INLINE __attribute__((noinline))

struct Vec128
{
    __m128 v;
};

// 标记为 SSE2 的 force inline 函数
SSE2_ATTR FORCE_INLINE Vec128 load(const float* mem) noexcept
{
    return { _mm_load_ps(mem) };
}

SSE2_ATTR FORCE_INLINE void store(float* mem, Vec128 v) noexcept
{
    _mm_store_ps(mem, v.v);
}

SSE2_ATTR FORCE_INLINE Vec128 add(Vec128 lhs, Vec128 rhs) noexcept
{
    return { _mm_add_ps(lhs.v, rhs.v) };
}

SSE2_ATTR FORCE_INLINE Vec128 mul_add_1(Vec128 a, Vec128 b, Vec128 c) noexcept
{
    return { _mm_add_ps(_mm_mul_ps(a.v, b.v), c.v) };
}

// AVX 函数
AVX_ATTR FORCE_INLINE Vec128 mul_add_2(Vec128 a, Vec128 b, Vec128 c) noexcept
{
    return { _mm_fmadd_ps(a.v, b.v, c.v) };
}

// kernel 是 AVX 的，测试是否生成VEX编码，有没有SSE到AVX指令的转换
AVX_ATTR NO_INLINE void kernel(
    const float* a,
    const float* b,
    const size_t size,
    float* out
) noexcept
{
    for (size_t i = 0; i < size; i += 4)
    {
        Vec128 aa = load(a + i);
        Vec128 bb = load(b + i);
        Vec128 result = mul_add_1(aa, bb, bb);
        Vec128 result2 = mul_add_2(aa, bb, aa);
        Vec128 result3 = add(result, result2);
        store(out + i, result3);
    }
}

#include <vector>

#include <xmmintrin.h>
#include <immintrin.h>

#include <kSimd/core/aligned_allocate.hpp>
#include <kSimd/core/impl/func_attr.hpp>

#include "test.hpp"


TEST(aligned_allocate, alignment_test)
{
    constexpr uintptr_t alignment = 32;

    for (int i = 1; i < 20; ++i)
    {
        float* mem = (float*)ksimd::aligned_allocate(i * sizeof(float) * 8, alignment);

        EXPECT_TRUE(reinterpret_cast<uintptr_t>(mem) % alignment == 0);

        ksimd::aligned_free(mem);
    }
}

TEST(aligned_allocate, std_vector)
{
    []() KSIMD_AVX2_MAX_INTRINSIC_ATTR
    {
        using Arr = std::vector<float, ksimd::AlignedAllocator<float>>;
        constexpr size_t size = 88;

        Arr numbers;
        for (size_t i = 0; i < size; ++i)
        {
            numbers.push_back(random_f(-2.0f, 2.0f));
        }

        float expected = 0.0f;
        for (size_t i = 0; i < size; ++i)
        {
            expected += numbers[i];
        }

        // simd test
        __m256 sum_8 = _mm256_setzero_ps();
        for (size_t i = 0; i < size; i += 8)
        {
            __m256 tmp = _mm256_load_ps(numbers.data() + i);
            sum_8 = _mm256_add_ps(sum_8, tmp);
        }

        // reduce sum
        // [8, 7, 6, 5, 4, 3, 2, 1]
        // hadd
        // [78, 56, 78, 56, 34, 12, 34, 12]
        // hadd
        // [5678, 5678, 5678, 5678, 1234, 1234, 1234, 1234]
        __m256 t1 = _mm256_hadd_ps(sum_8, sum_8);
        t1 = _mm256_hadd_ps(t1, t1);

        // low = [1234, 1234, 1234, 1234]
        // high = [5678, 5678, 5678, 5678]
        __m128 low = _mm256_castps256_ps128(t1);
        __m128 high = _mm256_extractf128_ps(t1, 0b1);

        // add
        // [12345678, ....]
        // get lane[0]
        low = _mm_add_ps(low, high);
        float result = _mm_cvtss_f32(low);

        EXPECT_NEAR(result, expected, 1e-4f);
    }();
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
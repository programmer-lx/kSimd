#include <iostream>

#include <kSimd/base_op.hpp>

#include "test.hpp"


#if defined(KSIMD_COMPILER_GCC) || defined(KSIMD_COMPILER_CLANG)
    #define TEST_FUNC_ATTR __attribute__((target("avx512f")))
#else
    #define TEST_FUNC_ATTR
#endif

TEST_FUNC_ATTR void call_avx512f()
{
    __m512 x = _mm512_set1_ps(1.0f);

    const volatile void* ptr = &x;
    std::cout << reinterpret_cast<uintptr_t>(ptr) << " ";

    float sum = _mm512_reduce_add_ps(x);
    std::cout << sum << std::endl;
}

void call_scalar()
{
    float x = 1;
    const volatile float* ptr = &x;
    std::cout << (*ptr) << std::endl;
}

using PFN = void(*)();
static inline PFN fn_arr[] = {
    call_scalar,
    call_avx512f
};

TEST(not_support, avx512_failed)
{
    const auto& result = ksimd::get_cpu_support_info();

    int idx = -1;
    // select in runtime
    if (result.AVX512_F)
    {
        idx = 1;
        std::cout << "support AVX512f" << std::endl;
    }
    else
    {
        idx = 0;
        std::cout << "NOT support AVX512f" << std::endl;
    }

    // call test
    auto fn = fn_arr[idx];
    fn();

    fn_arr[idx](); // also test this

    SUCCEED();
}


int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include <cassert>

#include <iostream>
#include <vector>

#include <kSimd/aligned_allocate.hpp>
#include <kSimd/simd_op.hpp>

int main()
{
    using trait = tsimd::SimdTraits<tsimd::SimdInstruction::SSE2, float>;
    using op = tsimd::SimdOp<tsimd::SimdInstruction::SSE2, float>;
    using batch_t = trait::batch_t;

    size_t alignment = tsimd::AlignedAllocator<float>::alignment();
    std::cout << "alignment = " << alignment << std::endl;

    std::vector<float, tsimd::AlignedAllocator<float>> a;
    std::vector<float, tsimd::AlignedAllocator<float>> b;
    std::vector<float, tsimd::AlignedAllocator<float>> c;

    constexpr size_t Total = 123;
    for (size_t i = 0; i < Total; ++i)
    {
        a.push_back(static_cast<float>(i));
        b.push_back(static_cast<float>(i + 2));
        c.push_back(static_cast<float>(i + 4));
    }

    // compute
    std::vector<float, tsimd::AlignedAllocator<float>> result(Total);
    constexpr size_t Step = trait::Lanes;
    size_t i = 0;
    for (; i + Step <= Total; i += Step)
    {
        batch_t a1 = op::load(a.data() + i);
        batch_t b1 = op::load(b.data() + i);
        batch_t c1 = op::load(c.data() + i);
        batch_t r = op::mul_add(a1, b1, c1);

        // store
        op::store(result.data() + i, r);
    }

    // compute the rest data
    for (; i < Total; ++i)
    {
        result[i] = a[i] * b[i] + c[i];
    }

    // test the result
    for (size_t j = 0; j < Total; ++j)
    {
        assert(result[j] == a[j] * b[j] + c[j]);
    }

    std::cout << "SUCCEED" << std::endl;

    return 0;
}
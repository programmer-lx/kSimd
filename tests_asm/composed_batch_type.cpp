#include <kSimd/base_op.hpp>

using namespace ksimd;

struct Float64x2x2
{
    static constexpr size_t Count = 2;
    x86_vector128::Batch<float64> v[Count];
};

KSIMD_OP_SSE2_API Float64x2x2 load(const float64* mem) noexcept
{
    return [&]<size_t... I>(std::index_sequence<I...>) -> Float64x2x2
    {
        return { BaseOp<SimdInstruction::SSE2, float64>::load(mem + I * 2)... };
    }(std::make_index_sequence<Float64x2x2::Count>{});
}

KSIMD_OP_SSE_API void store(float64* mem, Float64x2x2 v) noexcept
{
    return [&]<size_t... I>(std::index_sequence<I...>) -> void
    {
        (BaseOp<SimdInstruction::SSE2, float64>::store(mem + I * 2, v.v[I]), ...);
    }(std::make_index_sequence<Float64x2x2::Count>{});
}

KSIMD_OP_SSE2_API Float64x2x2 add(Float64x2x2 lhs, Float64x2x2 rhs) noexcept
{
    return [&]<size_t... I>(std::index_sequence<I...>) -> Float64x2x2
    {
        return { BaseOp<SimdInstruction::SSE2, float64>::add(lhs.v[I], rhs.v[I])... };
    }(std::make_index_sequence<Float64x2x2::Count>{});
}

KSIMD_OP_SSE2_API Float64x2x2 mul_add(Float64x2x2 a, Float64x2x2 b, Float64x2x2 c) noexcept
{
    return [&]<size_t... I>(std::index_sequence<I...>) -> Float64x2x2
    {
        return { BaseOp<SimdInstruction::SSE2, float64>::mul_add(a.v[I], b.v[I], c.v[I])... };
    }(std::make_index_sequence<Float64x2x2::Count>{});
}

void kernel(const float64* a, const float64* b, float64* out, const size_t size) noexcept
{
    for (size_t i = 0; i < size; i += 4)
    {
        Float64x2x2 a_v = load(a);
        Float64x2x2 b_v = load(b);
        Float64x2x2 result = add(a_v, b_v);
        result = mul_add(a_v, result, b_v);
        store(out + i, result);
    }
}

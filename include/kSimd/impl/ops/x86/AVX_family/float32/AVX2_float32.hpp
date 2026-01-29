#pragma once

#include "AVX_float32.hpp"

KSIMD_NAMESPACE_BEGIN

// AVX2与AVX的浮点运算指令一致
template<>
struct SimdOp<SimdInstruction::AVX2, float32> : SimdOp<SimdInstruction::AVX, float32>
{
    using traits = SimdTraits<SimdInstruction::AVX2, float32>;
    using batch_t = typename traits::batch_t;
    using scalar_t = typename traits::scalar_t;
};

KSIMD_NAMESPACE_END

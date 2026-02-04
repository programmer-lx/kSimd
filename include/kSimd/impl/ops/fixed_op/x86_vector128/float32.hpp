#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32>
{};

// AVX+ 继承SSE4.1的BaseOp，保持 batch_t 一致
// 对于某些函数，可以使用AVX指令重写
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32>
{};

KSIMD_NAMESPACE_END

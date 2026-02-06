#pragma once

#include "kSimd/impl/ops/base_op/x86_vector128/float32.hpp"
#include "kSimd/impl/ops/base_op/x86_vector256/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE, float32, 4, 2>
    : detail::Executor_SSE_float32<2>
    , FixedOpHelper<4>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE2, float32, 4, 2>
    : detail::Executor_SSE2_float32<2>
    , FixedOpHelper<4>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE3, float32, 4, 2>
    : detail::Executor_SSE3_float32<2>
    , FixedOpHelper<4>
{};

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSSE3, float32, 4, 2>
    : detail::Executor_SSSE3_float32<2>
    , FixedOpHelper<4>
{};


#define KSIMD_API(ret) KSIMD_OP_SSE4_1_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SSE4_1, float32, 4, 2>
    : detail::Executor_SSE4_1_float32<2>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm_dp_ps(a.v[0], b.v[0], imm8), _mm_dp_ps(a.v[1], b.v[1], imm8) };
    }
};
#undef KSIMD_API

// TODO 使用 m256 类型
// AVX+
#define KSIMD_API(ret) KSIMD_OP_AVX_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX, float32, 4, 2>
    : detail::Executor_AVX_float32<1>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm256_dp_ps(a.v[0], b.v[0], imm8) };
    }
};
#undef KSIMD_API

#define KSIMD_API(ret) KSIMD_OP_AVX2_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2, float32, 4, 2>
    : detail::Executor_AVX2_float32<1>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm256_dp_ps(a.v[0], b.v[0], imm8) };
    }
};
#undef KSIMD_API

#define KSIMD_API(ret) KSIMD_OP_AVX2_FMA3_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_AVX2_FMA3, float32, 4, 2>
    : detail::Executor_AVX2_FMA3_float32<1>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        constexpr int imm8 = (src_mask << 4) | dst_mask;
        return { _mm256_dp_ps(a.v[0], b.v[0], imm8) };
    }
};
#undef KSIMD_API

KSIMD_NAMESPACE_END

#pragma once

#include "kSimd/impl/ops/base_op/scalar/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN


#define KSIMD_API(ret) KSIMD_OP_SCALAR_API static ret KSIMD_CALL_CONV

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32, 4, 1>
    : BaseOpTraits_Scalar<float32, 1> // traits
    , detail::Executor_Scalar_FloatingPoint_Base<BaseOpTraits_Scalar<float32, 1>> // executor
    , FixedOpInfo<4, 1>
    , FixedOpHelper<4>
{
    KSIMD_API(batch_t) sequence() noexcept
    {
        return { 0, 1, 2, 3 };
    }

    KSIMD_API(batch_t) sequence(float32 base) noexcept
    {
        return { base, base + 1, base + 2, base + 3 };
    }

    KSIMD_API(batch_t) sequence(float32 base, float32 stride) noexcept
    {
        return { base, base + stride, base + 2 * stride, base + 3 * stride };
    }

    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        // 先求和
        float32 sum = 0.f;
        if constexpr (src_mask & X) sum += a.v[0] * b.v[0];
        if constexpr (src_mask & Y) sum += a.v[1] * b.v[1];
        if constexpr (src_mask & Z) sum += a.v[2] * b.v[2];
        if constexpr (src_mask & W) sum += a.v[3] * b.v[3];

        // 再分发结果
        batch_t res = { 0.f, 0.f, 0.f, 0.f };
        if constexpr (dst_mask & X) res.v[0] = sum;
        if constexpr (dst_mask & Y) res.v[1] = sum;
        if constexpr (dst_mask & Z) res.v[2] = sum;
        if constexpr (dst_mask & W) res.v[3] = sum;

        return res;
    }
};

#undef KSIMD_API


KSIMD_NAMESPACE_END

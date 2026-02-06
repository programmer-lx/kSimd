#pragma once

#include "kSimd/impl/ops/base_op/scalar/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN

#define KSIMD_API(ret) KSIMD_OP_SCALAR_API static ret KSIMD_CALL_CONV

template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32, 4, 2>
    : detail::Executor_Scalar_float32<2>
    , FixedOpInfo<4, 2>
    , FixedOpHelper<4>
{
    template<int src_mask, int dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        // 先求和
        float32 sum_0 = 0.f;
        float32 sum_1 = 0.f;

        if constexpr (src_mask & X) sum_0 += a.v[0] * b.v[0];
        if constexpr (src_mask & Y) sum_0 += a.v[1] * b.v[1];
        if constexpr (src_mask & Z) sum_0 += a.v[2] * b.v[2];
        if constexpr (src_mask & W) sum_0 += a.v[3] * b.v[3];

        if constexpr (src_mask & X) sum_1 += a.v[4] * b.v[4];
        if constexpr (src_mask & Y) sum_1 += a.v[5] * b.v[5];
        if constexpr (src_mask & Z) sum_1 += a.v[6] * b.v[6];
        if constexpr (src_mask & W) sum_1 += a.v[7] * b.v[7];

        // 再分发结果
        batch_t res = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

        if constexpr (dst_mask & X) res.v[0] = sum_0;
        if constexpr (dst_mask & Y) res.v[1] = sum_0;
        if constexpr (dst_mask & Z) res.v[2] = sum_0;
        if constexpr (dst_mask & W) res.v[3] = sum_0;

        if constexpr (dst_mask & X) res.v[4] = sum_1;
        if constexpr (dst_mask & Y) res.v[5] = sum_1;
        if constexpr (dst_mask & Z) res.v[6] = sum_1;
        if constexpr (dst_mask & W) res.v[7] = sum_1;

        return res;
    }
};

#undef KSIMD_API

KSIMD_NAMESPACE_END

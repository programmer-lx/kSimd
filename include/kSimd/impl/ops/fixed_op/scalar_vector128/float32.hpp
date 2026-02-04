#pragma once

#include "kSimd/impl/ops/base_op/scalar/float32.hpp"
#include "kSimd/impl/ops/fixed_op/FixedOp.hpp"

KSIMD_NAMESPACE_BEGIN


#define KSIMD_API(ret) KSIMD_OP_SCALAR_API static ret KSIMD_CALL_CONV
template<>
struct FixedOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32, 4>
    : BaseOp<SimdInstruction::KSIMD_DYN_INSTRUCTION_SCALAR, float32>
    , FixedOpHelper<4>
{
    template<uint8 src_mask, uint8 dst_mask>
    KSIMD_API(batch_t) dot(batch_t a, batch_t b) noexcept
    {
        (void)a;
        (void)b;
        return {  };
    }
};
#undef KSIMD_API


KSIMD_NAMESPACE_END

// #pragma once // do not use this, let it dispatch

// Elementary Math Functions

#include <cmath>
#include <utility>

#include "kSimd/simd_op.hpp"

namespace ksimd::ext::KSIMD_DYN_INSTRUCTION::math
{
    template<typename batch_t>
    KSIMD_EXT_SIG_DYN(batch_t, sin, (batch_t v))
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_SIMD_OP(scalar_t);
        constexpr SimdInstruction Instruction = op::CurrentInstruction;
        constexpr size_t Lanes = op::Lanes;

        if constexpr (Instruction == SimdInstruction::Scalar)
        {
            return [&]<size_t... I>(std::index_sequence<I...>) -> batch_t
            {
                return { std::sin(v.v[I])... };
            }(std::make_index_sequence<Lanes>{});
        }
        else
        {
            // TODO
            alignas(op::BatchAlignment) scalar_t V[Lanes]{};
            op::store(V, v);
            for (size_t i = 0; i < Lanes; ++i)
            {
                V[i] = std::sin(V[i]);
            }
            return op::load(V);
        }
    }
} // namespace ksimd::math

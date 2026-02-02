// do not use this, let it dispatch
// #pragma once

// Elementary Math Functions

#include <cmath>
#include <utility>

#include "kSimd/base_op.hpp"

#define KSIMD_EXT_MATH_INLINE_API(ret)   KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE ret KSIMD_CALL_CONV
#define KSIMD_EXT_MATH_FLATTEN_API(ret)  KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN ret KSIMD_CALL_CONV

namespace KSIMD_NAMESPACE_NAME::ext::KSIMD_DYN_INSTRUCTION::math
{
#pragma region ------------- any types -------------------------

    template<is_batch_type batch_t>
    KSIMD_EXT_MATH_FLATTEN_API(batch_t) clamp(batch_t v, batch_t min, batch_t max) noexcept
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_BASE_OP(scalar_t);

        return op::min(max, op::max(v, min));
    }

#pragma endregion ------------- any types -------------------------


#pragma region ------------- floating point -------------------------

    template<is_batch_type_includes<float32, float64> batch_t>
    KSIMD_EXT_MATH_FLATTEN_API(batch_t) lerp(batch_t a, batch_t b, batch_t t) noexcept
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_BASE_OP(scalar_t);

        // result = (b - a) * t + a
        // mul_add
        return op::mul_add(op::sub(b, a), t, a);
    }

    template<is_batch_type_includes<float32, float64> batch_t>
    KSIMD_EXT_MATH_FLATTEN_API(batch_t) sin(batch_t v) noexcept
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_BASE_OP(scalar_t);
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

#pragma endregion ------------- floating point -------------------------
} // namespace ksimd::math

#undef KSIMD_EXT_MATH_INLINE_API
#undef KSIMD_EXT_MATH_FLATTEN_API

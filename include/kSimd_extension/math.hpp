// do not use this, let it dispatch
// #pragma once

// Elementary Math Functions

#include <cmath>
#include <utility>

#include "kSimd/simd_op.hpp"

namespace KSIMD_NAMESPACE_NAME::ext::KSIMD_DYN_INSTRUCTION::math
{
#pragma region ------------- any types -------------------------

    #pragma region ------------- clamp -------------------------
    template<is_batch_type batch_t>
    KSIMD_EXT_SIG_DYN(batch_t, clamp, (batch_t v, batch_t min, batch_t max))
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_SIMD_OP(scalar_t);

        // result = max(min, min(v, max))
        return op::max(min, op::min(v, max));
    }

    template<is_scalar_type scalar_t>
    KSIMD_EXT_SIG_DYN(scalar_t, clamp, (scalar_t v, scalar_t min, scalar_t max))
    {
        return detail::max(min, detail::min(v, max));
    }
    #pragma endregion ------------- clamp -------------------------

    #pragma region ------------- safe_clamp -------------------------
    template<is_batch_type batch_t>
    KSIMD_EXT_SIG_DYN(batch_t, safe_clamp, (batch_t v, batch_t edge1, batch_t edge2))
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_SIMD_OP(scalar_t);

        // result = max(min, min(v, max))
        batch_t min = op::min(edge1, edge2);
        batch_t max = op::max(edge1, edge2);
        return op::max(min, op::min(v, max));
    }

    template<is_scalar_type scalar_t>
    KSIMD_EXT_SIG_DYN(scalar_t, safe_clamp, (scalar_t v, scalar_t edge1, scalar_t edge2))
    {
        scalar_t min = detail::min(edge1, edge2);
        scalar_t max = detail::max(edge1, edge2);
        return detail::max(min, detail::min(v, max));
    }
    #pragma endregion ------------- safe_clamp -------------------------

#pragma endregion ------------- any types -------------------------


#pragma region ------------- floating point -------------------------

    #pragma region ------------- lerp -------------------------
    template<is_batch_type_includes<float32, float64> batch_t>
    KSIMD_EXT_SIG_DYN(batch_t, lerp, (batch_t a, batch_t b, batch_t t))
    {
        using scalar_t = typename batch_t::scalar_t;
        using op = KSIMD_DYN_SIMD_OP(scalar_t);

        // result = a + (b - a) * t
        return op::add(a, op::mul(op::sub(b, a), t));
    }

    template<is_scalar_type_includes<float32, float64> scalar_t>
    KSIMD_EXT_SIG_DYN(scalar_t, lerp, (scalar_t a, scalar_t b, scalar_t t))
    {
        return a + (b - a) * t;
    }
    #pragma endregion ------------- lerp -------------------------

    #pragma region ------------- sin -------------------------
    template<is_batch_type_includes<float32, float64> batch_t>
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

    template<is_scalar_type_includes<float32, float64> scalar_t>
    KSIMD_EXT_SIG_DYN(scalar_t, sin, (scalar_t v))
    {
        return std::sin(v);
    }
    #pragma endregion ------------- sin -------------------------

#pragma endregion ------------- floating point -------------------------
} // namespace ksimd::math

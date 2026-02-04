// do not use this, let it dispatch
// #pragma once

// Elementary Math

#include <cmath>
#include <utility>

#include "kSimd_IDE/IDE_macros.hpp"

#define KSIMD_API(ret) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN ret KSIMD_CALL_CONV

// 由于模板参数op没有代码提示，所以给IDE一个默认值提示
#ifdef KSIMD_IDE
    #define KSIMD_IDE_HINT using op = BaseOp<SimdInstruction::AVX, float32>;
#else
    #define KSIMD_IDE_HINT
#endif

namespace KSIMD_NAMESPACE_NAME::ext::KSIMD_DYN_INSTRUCTION::vmath
{
#pragma region------------- any types -------------------------

    template<typename op, is_batch_type batch_t>
    KSIMD_API(batch_t) clamp(batch_t v, batch_t min, batch_t max) noexcept
    {
        KSIMD_IDE_HINT
        return op::min(max, op::max(v, min));
    }

#pragma endregion


#pragma region------------- floating point -------------------------

    template<typename op, is_batch_type_includes<float32, float64> batch_t>
    KSIMD_API(batch_t) lerp(batch_t a, batch_t b, batch_t t) noexcept
    {
        KSIMD_IDE_HINT
        // result = (b - a) * t + a
        return op::mul_add(op::sub(b, a), t, a);
    }

    template<typename op, is_batch_type_includes<float32, float64> batch_t>
    KSIMD_API(batch_t) sin(batch_t v) noexcept
    {
        KSIMD_IDE_HINT

        constexpr SimdInstruction Instruction = op::internal_instruction_;
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
            alignas(op::BatchAlignment) typename op::scalar_t V[Lanes]{};
            op::store(V, v);
            for (size_t i = 0; i < Lanes; ++i)
            {
                V[i] = std::sin(V[i]);
            }
            return op::load(V);
        }
    }

#pragma endregion
} // namespace KSIMD_NAMESPACE_NAME::ext::KSIMD_DYN_INSTRUCTION::vmath

#undef KSIMD_API

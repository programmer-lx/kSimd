#pragma once

#include "kSimd/impl/number.hpp"
#include "kSimd/impl/func_attr.hpp"

KSIMD_NAMESPACE_BEGIN

bool support_f16c_intrinsic() noexcept;

namespace detail
{
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)

    KSIMD_F16C_INTRINSIC_ATTR KSIMD_NOINLINE
    void f32_to_f16_F16C(
                float16* KSIMD_RESTRICT out,
        const   float32* KSIMD_RESTRICT in,
                size_t                  size
    ) noexcept;

#endif
}

inline void f32_to_f16(
            float16* KSIMD_RESTRICT out,
    const   float32* KSIMD_RESTRICT in,
    const   size_t                  size
) noexcept
{
    if (support_f16c_intrinsic()) [[likely]]
    {
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)
        detail::f32_to_f16_F16C(out, in, size);
#endif
    }
    else [[unlikely]]
    {
        for (size_t i = 0; i < size; ++i)
        {
            const float16 result = f32_to_f16(in[i]);
            out[i] = result;
        }
    }
}

KSIMD_NAMESPACE_END

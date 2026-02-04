#include "kSimd/impl/platform.hpp"
#include "kSimd/impl/number.hpp"
#include "kSimd/impl/func_attr.hpp"

KSIMD_NAMESPACE_BEGIN

namespace
{
    bool support_f16c_intrinsic_impl() noexcept
    {
        const CpuSupportInfo& info = get_cpu_support_info();
        return info.F16C;
    }
}

namespace detail
{
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)

    KSIMD_F16C_INTRINSIC_ATTR KSIMD_NOINLINE
    void f32_to_f16_F16C(
                float16* KSIMD_RESTRICT out,
        const   float32* KSIMD_RESTRICT in,
                size_t                  size
    ) noexcept
    {
        (void)out;
        (void)in;

        size_t i = 0;
        // float32 x 8
        for (; i + 8 <= size; i += 8)
        {
            // __m256 result = _mm256_cvtps_ph()
        }
    }

#endif
}

bool support_f16c_intrinsic() noexcept
{
    static bool result = support_f16c_intrinsic_impl();
    return result;
}

KSIMD_NAMESPACE_END

#include "kSimd/impl/ops/dispatch.hpp"

#include "kSimd/impl/utils.hpp"

KSIMD_NAMESPACE_BEGIN

namespace detail
{
    static int dyn_func_index_impl() noexcept
    {
        const auto& supports = get_cpu_support_info();

        // 从最高级的指令往下判断
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2) && defined(KSIMD_INSTRUCTION_FEATURE_FMA3) && defined(KSIMD_INSTRUCTION_FEATURE_F16C)
        if (supports.AVX2 && supports.FMA3 && supports.F16C)
        {
            return underlying(SimdInstructionIndex::AVX2_FMA3_F16C);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2)
        if (supports.AVX2)
        {
            return underlying(SimdInstructionIndex::AVX2);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX)
        if (supports.AVX)
        {
            return underlying(SimdInstructionIndex::AVX);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE2)
        if (supports.SSE2)
        {
            return underlying(SimdInstructionIndex::SSE2);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
        if (supports.SSE)
        {
            return underlying(SimdInstructionIndex::SSE);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
        return underlying(SimdInstructionIndex::Scalar);
#endif

        return 0;
    }
}

// 测试时，这段代码不会被调用
int dyn_func_index() noexcept
{
    static int i = detail::dyn_func_index_impl();
    return i;
}

KSIMD_NAMESPACE_END

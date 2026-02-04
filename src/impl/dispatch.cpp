#include "kSimd/impl/dispatch.hpp"

#include "kSimd/impl/utils.hpp"

KSIMD_NAMESPACE_BEGIN

namespace
{
    int dyn_func_index_impl() noexcept
    {
        const CpuSupportInfo supports = get_cpu_support_info();

        // 从最高级的指令往下判断
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_FMA3)
        if (supports.AVX2 && supports.FMA3)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX2_FMA3);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2)
        if (supports.AVX2)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX2);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX)
        if (supports.AVX)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE4_1)
        if (supports.SSE4_1)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SSE4_1);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSSE3)
        if (supports.SSSE3)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SSSE3);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE3)
        if (supports.SSE3)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SSE3);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE2)
        if (supports.SSE2)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SSE2);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
        if (supports.SSE)
        {
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SSE);
        }
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
        return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SCALAR);
#else
        return -1; // 某些平台可能不需要提供标量fallback
#endif
    }
}

// 测试时，这段代码不会被调用
int KSIMD_CALL_CONV dyn_func_index() noexcept
{
    static int i = dyn_func_index_impl();
    return i;
}

KSIMD_NAMESPACE_END

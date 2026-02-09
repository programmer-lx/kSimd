#include "kSimd/core/impl/dispatch.hpp"

#include "kSimd/core/impl/number.hpp"

namespace ksimd
{
    namespace
    {
        int dyn_func_index_impl() noexcept
        {
            const CpuSupportInfo& supports = get_cpu_support_info();

            // 从最高级的指令往下判断
            #if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
            if (supports.AVX2 && supports.FMA3 && supports.F16C)
            {
                return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_AVX2_MAX);
            }
            #endif

            // #if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
            //         return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_SCALAR);
            // #endif

            // 返回实际的 fallback index 即可，某些平台，标量可能不是 fallback
            return detail::underlying(detail::SimdInstructionIndex::KSIMD_DYN_INSTRUCTION_FALLBACK);
        }
    }

    // 测试时，这段代码不会被调用
    int KSIMD_CALL_CONV dyn_func_index() noexcept
    {
        static int i = dyn_func_index_impl();
        return i;
    }
}

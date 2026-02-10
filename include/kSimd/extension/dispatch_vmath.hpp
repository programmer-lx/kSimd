// do not use include guard

// 逐元素数学库

// check (必须在包含这个文件之前包含dispatch_core.hpp)
#include "kSimd/core/impl/ops/op_helpers.hpp"
#if !defined(KSIMD_DISPATCH_CORE_INCLUDED)
    static_assert(false, "We should include <kSimd/core/dispatch_core.hpp> before include <kSimd/extension/dispatch_vmath.hpp>");
#endif

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION::vmath
{
    // --- any type ---

    /**
     * @tparam option 是否严格检查NaN的传播 \n
     * 一般情况下，保持option是默认的Native即可，如果v的值是NaN，函数会保证NaN传播。\n
     * 如果觉得min,或max的值可能是NaN，可以将option设置为CheckNaN保证NaN的传播
     */
    template<OpHelper::FloatMinMaxOption option = OpHelper::FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) clamp(Batch<S> v, Batch<S> min, Batch<S> max) noexcept
    {
        using o = op<KSIMD_IDE_RUNTIME_TYPE_IDE_TYPE(S, float32)>;

        // 一定要将v放在右边，假如v是NaN，经过min之后，会返回NaN，然后经过了max之后，也会返回NaN
        // 所以一般情况下，保持option是Native即可。如果担心min,或max的值是NaN，可以将option设置为CheckNaN保证NaN的传播
        return o::max<option>(min, o::min<option>(max, v));
    }

    // --- floating point ---
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) lerp(Batch<S> a, Batch<S> b, Batch<S> t) noexcept
    {
        using o = op<KSIMD_IDE_RUNTIME_TYPE_IDE_TYPE(S, float32)>;

        // a + (b - a) * t
        return o::mul_add(o::sub(b, a), t, a);
    }
}

#undef KSIMD_API

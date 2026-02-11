// do not use include guard

// 逐元素数学库

// check (必须在包含这个文件之前包含dispatch_core.hpp)
#if !defined(KSIMD_DISPATCH_CORE_INCLUDED)
    #error We should include <kSimd/core/dispatch_core.hpp> before include <kSimd/extension/dispatch_vmath.hpp>
#endif

// #include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION::vmath
{
    // --- any type ---
    /**
     * @tparam option 是否严格检查NaN的传播 \n
     * 一般情况下，保持option是默认的Native即可，如果v的值是NaN，函数会保证NaN传播。\n
     * 如果觉得min_val或max_val的值可能是NaN，可以将option设置为CheckNaN保证NaN的传播
     */
    template<FloatMinMaxOption option = FloatMinMaxOption::Native, is_scalar_type S>
    KSIMD_API(Batch<S>) clamp(Batch<S> v, Batch<S> min_val, Batch<S> max_val) noexcept
    {
        // 一定要将v放在右边，假如v是NaN，经过min之后，会返回NaN，然后经过了max之后，也会返回NaN
        // 所以一般情况下，保持option是Native即可。如果担心min,或max的值是NaN，可以将option设置为CheckNaN保证NaN的传播
        return max<option>(min_val, min<option>(max_val, v));
    }

    // --- floating point ---
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) lerp(Batch<S> a, Batch<S> b, Batch<S> t) noexcept
    {
        // a + (b - a) * t
        return mul_add(sub(b, a), t, a);
    }
}

#undef KSIMD_API

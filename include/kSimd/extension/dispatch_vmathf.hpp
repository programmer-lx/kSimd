// do not use include guard

// 逐元素数学库

// check (必须在包含这个文件之前包含dispatch_this_file.hpp)
#if !defined(KSIMD_DETAIL_CHECK_DISPATCH_FILE_INCLUDED)
    static_assert(false, "KSIMD Error: <kSimd/core/dispatch_core.hpp> cannot be included directly."
    " Please include your dispatch header \"<kSimd/core/dispatch_this_file.hpp>\" and define KSIMD_DISPATCH_THIS_FILE.");
#endif

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION::vmathf
{
    template<is_scalar_floating_point S>
    KSIMD_API(Batch<S>) lerp(Batch<S> a, Batch<S> b, Batch<S> t) noexcept
    {
        // a + (b - a) * t
        return op<S>::mul_add(op<S>::sub(b, a), t, a);
    }
}

#undef KSIMD_API

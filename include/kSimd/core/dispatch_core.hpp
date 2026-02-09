// dot not use include guard

// check (必须在包含这个文件之前包含dispatch_this_file.hpp)
#if !defined(KSIMD_DETAIL_CHECK_DISPATCH_FILE_INCLUDED)
    static_assert(false, "KSIMD Error: <kSimd/core/dispatch_core.hpp> cannot be included directly."
    " Please include your dispatch header \"<kSimd/core/dispatch_this_file.hpp>\" and define KSIMD_DISPATCH_THIS_FILE.");
#endif

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if KSIMD_DYN_DISPATCH_LEVEL == KSIMD_DYN_DISPATCH_LEVEL_SCALAR
    #include "impl/ops/scalar.hpp"
#endif


// AVX family
#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_AVX_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_AVX_END
    #include "impl/ops/x86_avx_family.hpp"
#endif

// clang-format on

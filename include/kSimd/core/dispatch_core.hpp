// dot not use include guard

// check (必须在包含这个文件之前包含dispatch_this_file.hpp)
#if !defined(KSIMD_DETAIL_CHECK_DISPATCH_FILE_INCLUDED)
    #error <kSimd/core/dispatch_core.hpp> cannot be included directly. \
Please include your dispatch header "<kSimd/core/dispatch_this_file.hpp>" before and define KSIMD_DISPATCH_THIS_FILE.
#endif

// 用于检查是否已经包含了这个文件
#define KSIMD_DISPATCH_CORE_INCLUDED

// clang-format off

#include "impl/base.hpp"

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

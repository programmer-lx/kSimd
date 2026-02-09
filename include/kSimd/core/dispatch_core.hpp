// dot not use include guard

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

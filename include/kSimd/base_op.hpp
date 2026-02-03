#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/scalar/float32.hpp"
    #include "impl/ops/scalar/float64.hpp"
#endif


// SSE family
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY)
    #include "impl/ops/x86/SSE_family/float32.hpp"
    #include "impl/ops/x86/SSE_family/float64.hpp"
#endif


// AVX family
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)
    #include "impl/ops/x86/AVX_family/float32.hpp"
    #include "impl/ops/x86/AVX_family/float64.hpp"
#endif

// clang-format on

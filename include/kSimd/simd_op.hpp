#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar or SSE float64
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR) || defined(KSIMD_INSTRUCTION_FEATURE_SSE) // SSE不支持float64
    #include "impl/ops/Scalar/Scalar_float32.hpp"
    #include "impl/ops/Scalar/Scalar_float64.hpp"
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

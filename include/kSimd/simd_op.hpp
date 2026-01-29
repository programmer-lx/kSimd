#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/Scalar/Scalar_float32.hpp"
    #include "impl/ops/Scalar/Scalar_float64.hpp"
#endif


// SSE
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE)
    #include "impl/ops/x86/SSE_family/float32/SSE_float32.hpp"
    #include "impl/ops/x86/SSE_family/float64/SSE_float64.hpp"
#endif

// SSE2
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE2)
    #include "impl/ops/x86/SSE_family/float32/SSE2_float32.hpp"
    #include "impl/ops/x86/SSE_family/float64/SSE2_float64.hpp"
#endif

// SSE3
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE3)
    #include "impl/ops/x86/SSE_family/float32/SSE3_float32.hpp"
    #include "impl/ops/x86/SSE_family/float64/SSE3_float64.hpp"
#endif

// SSE4.1
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE4_1)
    #include "impl/ops/x86/SSE_family/float32/SSE4_1_float32.hpp"
    #include "impl/ops/x86/SSE_family/float64/SSE4_1_float64.hpp"
#endif


// AVX
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX)
    #include "impl/ops/x86/AVX_family/float32/AVX_float32.hpp"
    #include "impl/ops/x86/AVX_family/float64/AVX_float64.hpp"
#endif

// AVX2
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2)
    #include "impl/ops/x86/AVX_family/float32/AVX2_float32.hpp"
    #include "impl/ops/x86/AVX_family/float64/AVX2_float64.hpp"
#endif

// AVX2 + FMA3 + F16C
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2) && defined(KSIMD_INSTRUCTION_FEATURE_FMA3) && defined(KSIMD_INSTRUCTION_FEATURE_F16C)
    #include "impl/ops/x86/AVX_family/float32/AVX2_FMA3_F16C_float32.hpp"
    #include "impl/ops/x86/AVX_family/float64/AVX2_FMA3_F16C_float64.hpp"
#endif

// clang-format on

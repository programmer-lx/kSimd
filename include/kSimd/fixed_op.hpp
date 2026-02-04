#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
#endif


// SSE family
#if defined(KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY)
#endif


// AVX family
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)
#endif

// clang-format on

#pragma once

#include "impl/platform.hpp"

// clang-format off

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/Scalar/TypeOp.hpp"
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY)
    #include "impl/ops/x86/SSE_family/TypeOp.hpp"
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX512_FAMILY)
    #include "impl/ops/x86/AVX_family/TypeOp.hpp"
#endif

// clang-format on

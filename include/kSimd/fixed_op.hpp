#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/fixed_op/scalar_vector128/float32.hpp"
#endif


// x86
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)
    #include "impl/ops/fixed_op/x86_vector256/float32.hpp"
#endif

// clang-format on

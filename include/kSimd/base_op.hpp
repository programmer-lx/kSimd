#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/base_op/scalar/float32.hpp"
    #include "impl/ops/base_op/scalar/float64.hpp"
#endif


// AVX family
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX_FAMILY)
    #include "impl/ops/base_op/x86_vector256/float32.hpp"
    #include "impl/ops/base_op/x86_vector256/float64.hpp"
#endif

// clang-format on

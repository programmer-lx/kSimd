#pragma once

// clang-format off

#include "impl/platform.hpp"

// Scalar
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/fixed_op/scalar/float32_4x1.hpp"
#endif


// x86
#if defined(KSIMD_ARCH_X86_ANY)
    #include "impl/ops/fixed_op/x86/float32_4x2.hpp"
#endif

// clang-format on

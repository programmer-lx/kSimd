#pragma once

#include "impl/platform.hpp"

// clang-format off

#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #include "impl/ops/type_op/scalar.hpp"
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_SSE_FAMILY)
    #include "impl/ops/type_op/x86_vector128.hpp"
#endif

#if defined(KSIMD_INSTRUCTION_FEATURE_AVX512_FAMILY)
    #include "impl/ops/type_op/x86_vector256.hpp"
#endif

// clang-format on

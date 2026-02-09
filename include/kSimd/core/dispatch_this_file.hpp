#pragma once

#include "impl/dispatch.hpp"
#include "impl/func_attr.hpp"

#if !defined(KSIMD_DISPATCH_THIS_FILE)
    #error "has not defined KSIMD_DISPATCH_THIS_FILE"
#endif


#undef KSIMD_ONCE
#define KSIMD_ONCE 0

// AVX2 + FMA3 + F16C
#if defined(KSIMD_INSTRUCTION_FEATURE_AVX2_MAX)
    #undef KSIMD_DYN_INSTRUCTION
    #define KSIMD_DYN_INSTRUCTION KSIMD_DYN_INSTRUCTION_AVX2_MAX

    // 此时 KSIMD_DYN_FUNC_ATTR 等于 AVX2_FMA3
    #undef KSIMD_DYN_FUNC_ATTR
    #define KSIMD_DYN_FUNC_ATTR KSIMD_AVX2_MAX_INTRINSIC_ATTR

    #include KSIMD_DISPATCH_THIS_FILE
#endif

// Scalar (may be fallback)
#if defined(KSIMD_INSTRUCTION_FEATURE_SCALAR)
    #undef KSIMD_DYN_INSTRUCTION
    #define KSIMD_DYN_INSTRUCTION KSIMD_DYN_INSTRUCTION_SCALAR

    // 此时 KSIMD_DYN_FUNC_ATTR 等于 Scalar
    #undef KSIMD_DYN_FUNC_ATTR
    #define KSIMD_DYN_FUNC_ATTR KSIMD_SCALAR_INTRINSIC_ATTR

    #if (KSIMD_INSTRUCTION_FEATURE_SCALAR != KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE)
        #include KSIMD_DISPATCH_THIS_FILE // dispatch if not fallback
    #endif
#endif


// last dispatch
// once
#undef KSIMD_ONCE
#define KSIMD_ONCE 1

#undef KSIMD_DISPATCH_THIS_FILE

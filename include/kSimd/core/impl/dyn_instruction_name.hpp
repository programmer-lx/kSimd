#pragma once

#include "platform.hpp"

// --------------------------------- KSIMD_DYN_INSTRUCTION names ---------------------------------
#define KSIMD_DYN_INSTRUCTION_SCALAR   SCALAR
#define KSIMD_DYN_INSTRUCTION_AVX2_MAX AVX2_MAX


// fallback instruction

// scalar fallback
#if KSIMD_INSTRUCTION_FEATURE_SCALAR == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #undef KSIMD_DYN_INSTRUCTION_FALLBACK
    #define KSIMD_DYN_INSTRUCTION_FALLBACK KSIMD_DYN_INSTRUCTION_SCALAR
#endif

// check fallback
#if !defined(KSIMD_DYN_INSTRUCTION_FALLBACK)
    #error "We must define a fallback instruction name."
#endif

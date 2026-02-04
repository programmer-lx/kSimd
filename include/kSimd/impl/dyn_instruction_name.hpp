#pragma once

#include "platform.hpp"

// --------------------------------- KSIMD_DYN_INSTRUCTION names ---------------------------------
#define KSIMD_DYN_INSTRUCTION_SCALAR            Scalar
#define KSIMD_DYN_INSTRUCTION_SSE               SSE
#define KSIMD_DYN_INSTRUCTION_SSE2              SSE2
#define KSIMD_DYN_INSTRUCTION_SSE3              SSE3
#define KSIMD_DYN_INSTRUCTION_SSSE3             SSSE3
#define KSIMD_DYN_INSTRUCTION_SSE4_1            SSE4_1
#define KSIMD_DYN_INSTRUCTION_SSE4_2            SSE4_2
#define KSIMD_DYN_INSTRUCTION_AVX               AVX
#define KSIMD_DYN_INSTRUCTION_AVX2              AVX2
#define KSIMD_DYN_INSTRUCTION_AVX2_FMA3         AVX2_FMA3


// fallback instruction

// scalar fallback
#if KSIMD_INSTRUCTION_FEATURE_SCALAR == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #undef KSIMD_DYN_INSTRUCTION_FALLBACK
    #define KSIMD_DYN_INSTRUCTION_FALLBACK KSIMD_DYN_INSTRUCTION_SCALAR
#endif

// SSE2 fallback
#if KSIMD_INSTRUCTION_FEATURE_SSE2 == KSIMD_INSTRUCTION_FEATURE_FALLBACK_VALUE
    #undef KSIMD_DYN_INSTRUCTION_FALLBACK
    #define KSIMD_DYN_INSTRUCTION_FALLBACK KSIMD_DYN_INSTRUCTION_SSE2
#endif

// check fallback
#if !defined(KSIMD_DYN_INSTRUCTION_FALLBACK)
    #error "We must define a fallback instruction name."
#endif

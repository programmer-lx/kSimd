#pragma once

#include "base.hpp"

// see https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html#x86-Function-Attributes for more intrinsics
// attributes information

// scalar
#define KSIMD_INTRINSIC_ATTR_SCALAR
#define KSIMD_OP_SCALAR_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_INTRINSIC_ATTR_SCALAR


// avx2+fma3+f16c
#define KSIMD_INTRINSIC_ATTR_AVX2_MAX KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2,fma,f16c")
#define KSIMD_OP_AVX2_FMA3_F16C_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_INTRINSIC_ATTR_AVX2_MAX

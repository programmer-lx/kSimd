#pragma once

#include "platform.hpp"

// see https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html#x86-Function-Attributes for more intrinsics attributes information

// scalar
#define KSIMD_SCALAR_INTRINSIC_ATTR
#define KSIMD_OP_SCALAR_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_SCALAR_INTRINSIC_ATTR


// avx2+fma3
#define KSIMD_AVX2_FMA3_F16C_INTRINSIC_ATTR KSIMD_FUNC_ATTR_INTRINSIC_TARGETS("avx2,fma,f16c")
#define KSIMD_OP_AVX2_FMA3_F16C_API \
    KSIMD_FORCE_INLINE \
    KSIMD_FLATTEN \
    KSIMD_AVX2_FMA3_F16C_INTRINSIC_ATTR

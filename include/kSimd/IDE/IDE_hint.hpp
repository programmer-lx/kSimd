#pragma once

#if defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__)
    #define KSIMD_IDE 1
#endif

#if KSIMD_IDE

    #include "kSimd/core/dispatch_this_file.hpp"
    #include "kSimd/core/dispatch_core.hpp"

    #define KSIMD_IDE_RUNTIME_TYPE_IDE_TYPE(runtime_type, ide_type) ide_type

#else

    #define KSIMD_IDE_RUNTIME_TYPE_IDE_TYPE(runtime_type, ide_type) runtime_type

#endif

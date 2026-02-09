#pragma once

#if defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__)
    #define KSIMD_IDE 1
#endif

#if KSIMD_IDE
    #include "kSimd/core/dispatch_this_file.hpp"
    #include "kSimd/core/dispatch_core.hpp"
#endif

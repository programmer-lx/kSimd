#pragma once

// 如果没有提示 KSIMD_DYN_INSTRUCTION 宏，可以临时包含这个头文件，提供一些提示，不过开发完成后，记得注释掉，
// 否则会因为多次包含 dispatch_this_file.hpp 文件，而导致IDE报红

#if defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__)
    #define KSIMD_IDE 1
#endif

#if KSIMD_IDE
    #include "kSimd/core/dispatch_this_file.hpp"
    #include "kSimd/core/dispatch_core.hpp"
#endif

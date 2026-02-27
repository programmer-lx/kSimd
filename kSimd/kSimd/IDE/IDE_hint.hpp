#pragma once

// 这个文件主要用于“欺骗”IDE，重定义一些动态分发时才会定义的宏
// 这个文件一定要包含在所有的头文件的下面

#if !defined(KSIMD_IDE) && \
    (defined(__JETBRAINS_IDE__) || defined(__CLION_IDE__) || defined(__INTELLISENSE__))
    #define KSIMD_IDE 1
#endif

#if KSIMD_IDE

    // 可以自己取消注释，定义这些宏，方便开发测试
    // #undef KSIMD_DYN_INSTRUCTION
    // #define KSIMD_DYN_INSTRUCTION KSIMD_IDE_NS

    // #undef KSIMD_DYN_FUNC_ATTR
    // #define KSIMD_DYN_FUNC_ATTR

    // #undef KSIMD_DYN_DISPATCH_LEVEL
    // #define KSIMD_DYN_DISPATCH_LEVEL KSIMD_DYN_DISPATCH_LEVEL_X86_V2

#endif

# 功能宏

- `KSIMD_DYN_FUNC_ATTR`: 每个被分发的函数，都要加上这个宏，否则可能无法正确编译
- `KSIMD_ONCE`: 在这个宏内的所有代码，只会被编译一次。加上这个宏，防止文件递归包含的时候，出现符号重定义的情况
- `KSIMD_RESTRICT`: 适用于指针函数参数，用于提示编译器优化

# 调试宏
- `KSIMD_DEBUG_ENABLE_BASELINE_MESSAGE` : 每个编译单元编译的时候，将会输出当前的指令集下限

# 用于取消某条分发路径
- `KSIMD_DISABLE_X86_V4` : 取消 AVX512 F, DQ, VL 的分发
- `KSIMD_DISABLE_X86_V3` : 取消 AVX, AVX2, FMA3, F16C 的分发
- `KSIMD_DISABLE_X86_V2` : 取消 SSE, SSE2, SSE3, SSSE3, SSE4.1 的分发
- `KSIMD_DISABLE_NEON`   : 取消 NEON 的分发

# 判断当前分发到了哪个路径
`KSIMD_DYN_DISPATCH_LEVEL`: 如无特殊情况，最好不要使用这个宏。
如果有特殊需求，需要知道当前是不是分发到了某些指令集，想要执行特殊逻辑的时候，才使用这个宏来判断

- `KSIMD_DYN_DISPATCH_LEVEL_SSE_START`, `KSIMD_DYN_DISPATCH_LEVEL_SSE_END`: 当前正在分发SSE家族指令。
示例:
```c++
#if KSIMD_DYN_DISPATCH_LEVEL > KSIMD_DYN_DISPATCH_LEVEL_SSE_START && \
    KSIMD_DYN_DISPATCH_LEVEL < KSIMD_DYN_DISPATCH_LEVEL_SSE_END
    // do something...
#endif
```

- `KSIMD_DYN_DISPATCH_LEVEL_AVX_START`, `KSIMD_DYN_DISPATCH_LEVEL_AVX_END`: 当前正在分发AVX家族的指令
- `KSIMD_DYN_DISPATCH_LEVEL_AVX512_START`, `KSIMD_DYN_DISPATCH_LEVEL_AVX512_END`: 当前正在分发AVX512家族的指令
- `KSIMD_DYN_DISPATCH_LEVEL_X86_V2`: 当前正在分发 X86-64 V2 指令集
- `KSIMD_DYN_DISPATCH_LEVEL_X86_V3`: 当前正在分发 X86-64 V3 指令集
- `KSIMD_DYN_DISPATCH_LEVEL_X86_V4`: 当前正在分发 X86-64 V4 指令集
- `KSIMD_DYN_DISPATCH_LEVEL_NEON`: 当前正在分发 NEON 指令集

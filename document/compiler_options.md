# 编译参数

# 编译参数对指令集的下限的影响
如果开发者定义了某些编译参数，比如 -msse4.1，那么意味着分发scalar已经没有意义了，所以库将会对这些编译参数进行判断，从而在编译期裁剪分发表，减小二进制的体积。
分发表裁剪不是全局的，而是针对每个编译单元的。开发者可以给某个cpp文件开启特殊的编译参数，来实现编译单元粒度的分发表裁剪。

## GCC/clang
### X86
详见文档: https://gcc.gnu.org/onlinedocs/gcc-15.2.0/gcc/x86-Options.html
- 如果只想分发高于等于 X86-64-V4 的指令集，开启: `-mavx512f -mavx512dq -mavx512vl` 或 `-march=x86-64-v4`
- 高于或等于 X86-64-V3 : `-mavx2 -mfma -mf16c` 或 `-march=x86-64-v3`
- 高于或等于 X86-64-V2 : `-msse4.1` 或 `-march=x86-64-v2`

### ARM
TODO

## MSVC
### X86
详见文档: https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
- 高于或等于 X86-64-V4 : `/arch:AVX512` 或 `/arch:AVX10.1`(Visual Studio 17.13 +) 或 `/arch:AVX10.2`(Visual Studio 2026 +)
- 高于或等于 X86-64-V3 : 由于MSVC没有 \_\_F16C\_\_ 宏，所以这条路径跳过
- 高于或等于 X86-64-V2 : `/arch:AVX`

### ARM
TODO

## 小提示
如果开启了 `KSIMD_DEBUG_ENABLE_BASELINE_MESSAGE` 宏，每个编译单元在编译的时候，都会使用 #pragma message 输出当前的 baseline 指令集，在编译的时候可以查看

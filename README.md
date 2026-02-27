# kSimd (SIMD运行时分发库)

# 2026年Steam硬件软件支持调查

https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam?l=schinese

数据显示，20%的用户支持AVX512F，95%的用户支持AVX2, FMA3, F16C，98%的用户支持SSE4.1。

目前，库的分发路径如下:

- x86:
  1. AVX512-F + AVX512-DQ + AVX512-VL (X86-64 V4)
  2. AVX + AVX2 + FMA3 + F16C (X86-64 V3)
  3. SSE + SSE2 + SSE3 + SSSE3 + SSE4.1 (X86-64 V2)
  4. Scalar (fallback)
- arm:
  1. NEON
  2. Scalar (fallback)

# 文档
[简介](./document/introduction.md)
[功能宏](./document/macros.md)
[编译参数](./document/compiler_options.md)

# 第三方库
FP16: 用于标量的FP16转换: https://github.com/Maratyszcza/FP16
[许可证](./3rdparty/FP16/LICENSE)

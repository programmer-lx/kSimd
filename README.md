# kSimd (SIMD运行时分发库)

# 2026年Steam硬件软件支持调查
https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam?l=schinese

数据显示，95%以上的用户支持AVX2 + FMA3 + F16C，所以决定不再开发SSE家族的指令抽象。
目前，kSimd库支持两个分发路径: 1. AVX2+FMA3+F16C; 2. Scalar (fallback)

由于AVX512只有20%的用户支持，所以将AVX512的开发周期延后。

# 第三方库
FP16: 用于标量的FP16转换: https://github.com/Maratyszcza/FP16

[许可证](./3rdparty/FP16/LICENSE)

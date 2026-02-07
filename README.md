# kSimd (SIMD运行时分发库，用于自定义 SIMD kernel)

# 目标
1. 抽象SIMD指令，支持一次性编写kernel函数，生成多个指令版本，并且能够运行时根据CPUID选择最优指令集。
2. 支持对同一块内存的多重解释，比如将 256bit 内存解释为 float32 8x1 和 float32 4x2，对 float32 8x1 做 dot 运算，就是对8个元素进行 dot 运算；对 float32 4x2 做 dot 运算，就是分别对两个 float32x4 做 dot 运算。

# 2026年Steam硬件软件支持调查
https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam?l=schinese

数据显示，95%以上的用户支持AVX2 + FMA3 + F16C，所以决定不再开发SSE家族的指令抽象。
目前，kSimd库支持两个分发路径: 1. AVX2+FMA3+16C; 2. Scalar (fallback)

由于AVX512只有20%的用户支持，所以将AVX512的开发周期延后。

# Types

SIMD数据类型命名为 Batch，Mask类型命名为 Mask。
Batch和Mask类型封装了一个原生SIMD类型的C数组，比如：

```C++
template<scalar_type s, size_t reg_count>
struct Batch
{
    __m256 v[reg_count];
};
template<scalar_type s, size_t reg_count>
struct Mask
{
    __m256 m[reg_count];
    // __mmask8 m[reg_count]; // or AVX-512 mask type
};
```

# Op

## Executor (在detail命名空间内)
Executor的作用是：封装所有的SIMD垂直指令，在不知道 reg_count 的情况下，对每个 element 做相同的运算。

1. 每个函数使用 std::index_sequence 将循环展开。
2. 涉及到水平的操作，必须写进 mixin，比如 reduce_add, sequence。
3. mask操作不能直接写进 executor，比如独立成 mixin。因为AVX512有独立的 mask 类型，AVX512之前则用 __mXXX 类型代替。

## BaseOp\<Instruction, ScalarType\>
1. 直接继承 Executor ，继承其所有垂直操作，并特化 Executor 的 reg_count为 1
2. 继承相关的 mixin 类，补充 reg_count == 1 的水平操作和mask操作

## FixedOp\<Instruction, ScalarType, Width, Count\>
Instruction: 指令集，不需要自动填写，在动态分发的过程中，由宏帮忙填写

Width：一批数据的宽度
Count：有多少批数据
比如FixedOp\<AVX2, float32, 4, 2\>表示 float32 4x2，如果调用 dot 函数，就意味着对高128bit和低128bit分别点乘。

FixedOp用于操作固定Lanes的SIMD类型，可支持多种水平操作，以及对同一块内存的多种理解，比如float32 8x1, float32 4x2等。

继承Executor，在垂直操作基础之上，增加水平操作。

## TypeOp
用于跨类型操作，比如convert, bit_cast等。

# 第三方库
FP16: 用于标量的FP16转换: https://github.com/Maratyszcza/FP16

[许可证](./3rdparty/FP16/LICENSE)

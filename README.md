# kSimd (SIMD运行时分发库，用于自定义 SIMD kernel)

# 目标
1. 抽象SIMD指令，支持一次性编写kernel函数，生成多个指令版本，并且能够运行时根据CPUID选择最优指令集。
2. 支持对同一块内存的多重解释，比如将 256bit 内存解释为 float32 8x1 和 float32 4x2，对 float32 8x1 做 dot 运算，就是对8个元素进行 dot 运算；对 float32 4x2 做 dot 运算，就是分别对两个 float32x4 做 dot 运算。

# Types

SIMD数据类型命名为 Batch，Mask类型命名为 Mask。
Batch和Mask类型封装了一个原生SIMD类型的C数组，比如：

```C++
template<scalar_type s, size_t reg_count>
struct Batch
{
    __m128 v[reg_count];
};
template<scalar_type s, size_t reg_count>
struct Mask
{
    __m128 m[reg_count];
    // __mmask8 m[reg_count]; // or AVX-512 mask type
};
```

之所以是数组，是因为实际开发中可能会出现这种情况：
我有一堆的Vector4需要计算，但是AVX的寄存器宽度为256位，可以一次性读取2个Vector4，如果仅仅将这片内存视为float32x8，那么计算起来就不太方便，如果直接限制寄存器宽度为128位，那白白浪费了计算机的性能。所以需要有 float32 8x1 和 float32 4x2 等数据类型，来支持我使用256位指令来完成多种操作。
然而SSE家族只支持128位，只能加载1个Vector4。为了适配AVX家族 float32 4x2 的操作，必须要将两个 __m128 封装在一起，否则在运行时分发的过程中，将会无法编译通过。

然而CPU的指令集所支持的位宽不断上涨，比如AVX512可以一次性计算16个float32。所以将 reg_count 变为模板参数。



# Op

## Executor (在detail命名空间内)
Executor的作用是：封装所有的SIMD垂直指令，在不知道 reg_count 的情况下，对每个 element 做相同的运算。

每个函数使用 std::index_sequence 将循环展开。

## BaseOp\<Instruction, ScalarType\>
直接继承 Executor ，继承其所有垂直操作，并特化 Executor 的 reg_count为 1

## FixedOp\<Instruction, ScalarType, Width, Count\>
Instruction: 指令集，不需要自动填写，在动态分发的过程中，由宏帮忙填写

Width：一批数据的宽度
Count：有多少批数据
比如FixedOp\<AVX2, float32, 4, 2\>表示 float32 4x2，如果调用 dot 函数，就意味着对高128bit和低128bit分别点乘。

FixedOp用于操作固定Lanes的SIMD类型，可支持多种水平操作，以及对同一块内存的多种理解，比如float32 8x1, float32 4x2等。

FixedOp继承自Executor，在垂直操作基础之上，增加水平操作。

## TypeOp
用于跨类型操作，比如convert, bit_cast等。

# 设计原则
## 类继承
每一个 SimdOp\<Instruction, ...\> 只能编写 **<= Instruction** 的指令(SSE op 中不能含有SSE2指令)。
SSE2 op 要继承 SSE op，来复用比他更加低级的指令，以及使用更高级的指令覆盖掉父类的低级指令。

## 函数
op类不能出现任何的虚函数，全部函数必须 force inline。

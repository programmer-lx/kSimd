# ops
这个文件夹是对基础指令的封装，不涉及高级数学运算。
高级数学运算放到 extension 模块

# SIMD Vector Type
每个SIMD类型命名为 Batch，Mask 类型命名为 Mask。
Batch和Mask类型封装了一个原生SIMD类型的C数组，比如：
```C++
template<scalar_type s, size_t reg_count>
struct Batch
{
    __m128 v[reg_count];
};
```
之所以是数组，是因为实际开发中可能会出现这种情况：
我有一堆的Vector4需要计算，但是AVX的寄存器宽度为256位，可以一次性读取2个Vector4，如果仅仅将这片内存视为float32x8，
那么计算起来就不太方便，所以需要有 float32 8 和 float32 4x2 等数据类型。
然而SSE家族的寄存器宽度为128位，只能加载1个Vector4。为了适配AVX家族 float32 4x2 的操作，所以必须要将两个 __m128 封装在一起。

# Op

## Executor (在detail命名空间内)
封装所有垂直指令。
使用 std::index_sequence 展开。
编写这一层，是为了之后的多个SIMD类型的相同运算做准备，比如一个256位SIMD类型，我可以理解为 float32 8，
也可以理解为 float32 4x2，如果没有这一层的模板展开，则之后需要编写大量重复的垂直操作代码。

## BaseOp
直接继承 Executor ，继承其所有垂直操作，特化 Executor 的 RegCount 为 1

## TypeOp
用于跨类型操作，比如convert, bit_cast等。

## FixedOp
操作固定Lanes的SIMD类型，可支持多种水平操作。
FixedOp继承自BaseOp，在垂直操作基础之上，增加水平操作

# 设计原则
## 类继承
每一个 SimdOp<Instruction, ScalarType> 只能编写 **<= Instruction** 的指令(SSE op 中不能含有SSE2指令)。
SSE2 op 要继承 SSE op，来复用比他更加低级的指令，以及使用更高级的指令覆盖掉父类的低级指令。

## 函数
op类不能出现任何的虚函数，全部函数必须 force inline。

// do not use include guard

#include "op.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

/*

SIMD抽象封装，需要有以下功能：
1.  逐元素批量加速运算：
    此类操作不涉及水平操作，需要硬件的最宽SIMD宽度来使吞吐量最大
    使用 FullTag<ScalarType> 作为函数的第一个参数

2.  数学向量操作：
    (1) 可以使用 load_interleaved 和 store_deinterleaved 进行类似于矩阵转置的方式，进行加载和存储，将Aos变成SoA，这样就可以使用Full逻辑来操作。
    (2) 如果有一堆N维向量存储在一个连续的buffer中，我们可以通过类似于矩阵转置的方式，将AoS布局在load的时候变为SoA，在store的时候变回AoS。
        但是有些时候可能会有特殊情况，无法进行转置。所以需要设置一个专门的Fixed128Tag<ScalarType>来处理这种特殊情况。

3.  图像处理中，临近元素的操作：
    有一段buffer: [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
                 [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
                 [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
                 [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
                 [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    这个buffer存储结构是一个一维向量，但是逻辑结构是二维的。我们需要对他进行卷积运算，需要划分一个5x5的区域。
    这种操作，仍然使用Full来处理即可，使用loadu非对齐加载或者进行寄存器偏移来处理即可

4.  bit_cast：
    对同一块内存的重新解释。所以对于标量来说，我们仍然需要模拟SIMD，我们将标量模拟为64位的SIMD类型

5.  static_cast：类型转换
    (1) 位宽提升：u32 -> f64。一般来说，这种类型转换的操作应该要在load和store之前完成。在这个时候，我们可以构造一个HalfFullTag去进行load，
        这样就相当于加载Full的一半，然后我们调用promote函数，将类型进行提升，这样就刚好能够装满。
        如果是是 u8 -> u64 这种，可以使用 HalfFullTag(HalfFullTag) 连续构造两次，这样子，load函数就只会load 1/4，然后再用相同的Tag
        进行promote，就能刚好填满一个Full的寄存器
    (2) 位宽下降：u64 -> f32。一般位宽下降的操作，是在load/store之前完成的。

6.  reduce，水平规约：
    提供累加求和，累加乘法，min, max操作，返回一个标量


所以提供以下几种Tag:
FullTag<ScalarType> 表示吞吐量最大化，用满SIMD寄存器宽度
HalfFullTag<ScalarType> 只用于 load/store 的特殊tag，可以递归构造，每次长度减半
Fixed128Tag<ScalarType> 无论机器支持的寄存器位宽有多大，在定长向量的CPU中，始终使用128bit，对于变长向量(比如SVE)，始终使用前128位

*/

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
    // tags
    template<is_scalar_type S>
    struct FullTag : Tag_base<S, TagType::FullTag> {};

    template<is_scalar_type S>
    struct HalfFullTag : Tag_base<S, TagType::HalfFullTag>
    {
        static_assert(false, "TODO: HalfFullTag");
    };

    template<is_scalar_type S>
    struct Fixed128Tag : Tag_base<S, TagType::Fixed128Tag>
    {
        static_assert(false, "TODO: Fixed128Tag");
    };

    enum class RoundingMode
    {
        Up,         // 向上取整
        Down,       // 向下取整
        Nearest,    // 向最近偶数取整
        ToZero,     // 向0取整
        Round       // 四舍五入
    };

    enum class FloatMinMaxOption
    {
        Native,     // 按照原生硬件指令的行为
        CheckNaN    // 检查NaN的传播 (如果传入的值有一个NaN，则会返回NaN)
    };
}

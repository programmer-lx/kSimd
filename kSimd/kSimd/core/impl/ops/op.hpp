#pragma once

#include "kSimd/core/impl/types.hpp"

/*

SIMD抽象封装，需要有以下功能：
1.  逐元素批量加速运算：
    此类操作不涉及水平操作，需要硬件的最宽SIMD宽度来使吞吐量最大
    使用 FullTag<ScalarType> 作为函数的第一个参数

2.  数学向量操作：
    (1) 可以使用 load_interleaved 和 store_deinterleaved 进行类似于矩阵转置的方式，进行加载和存储，将Aos变成SoA，这样就可以使用Full逻辑来操作。
    (2) 有些时候可能会有特殊情况，无法使用interleaved处理向量。所以需要设置一个专门的Fixed128Tag<ScalarType>，提供水平操作，来处理这种特殊情况。
        之所以选择Fixed128 bit，是因为目前所有机器的最低向量宽度就是128bit。像 X86 SSE,AVX 这种定长向量，可以使用数组来模拟“超额”的宽度，
        但是SVE是变长向量，无法作为结构体的数据成员，如果在不支持SVE-256bit的电脑上，强行模拟256bit，会非常困难，所以选择128bit。

3.  图像处理中，临近元素的操作：
    有一段buffer:
    [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    [ R|G|B|A, R|G|B|A, R|G|B|A|, R|G|B|A, R|G|B|A ]
    这个buffer存储结构是一个一维向量，但是逻辑结构是二维的。我们需要对他进行卷积运算，需要划分一个5x5的区域。
    这种操作，仍然使用Full来处理即可，使用loadu非对齐加载或者进行寄存器偏移来处理即可

4.  bit_cast：
    对同一块内存的重新解释。所以对于标量来说，我们仍然需要模拟SIMD，我们将标量模拟为64位的SIMD类型

5.  static_cast：类型转换
    (1) 位宽提升：u32 -> f64。一般来说，这种类型转换的操作应该要在load和store之前完成。在这个时候，我们可以构造一个HalfIoTag去进行load，
        这样就相当于加载Full的一半，然后我们调用promote函数，将类型进行提升，这样就刚好能够装满。
        如果是是 u8 -> u64 这种，可以使用 auto tag = HalfIoTag(HalfIoTag()); 连续构造两次，这样子，load函数就只会load 1/4，然后
        进行promote，就能刚好填满一个Full的寄存器。
    (2) 位宽下降：u64 -> f32。一般位宽下降的操作，是在load/store之前完成的，使用demote函数，拼接多个高位宽的数据，返回一个低位宽的变量

6.  reduce，水平规约：
    提供累加求和，累加乘法，min, max操作，返回一个标量


所以提供以下几种Tag:
FullTag<ScalarType> 表示吞吐量最大化，用满SIMD寄存器宽度
HalfIoTag<ScalarType> 只用于 load/store 的特殊tag，可以递归构造，每次长度减半
Fixed128Tag<ScalarType> 无论机器支持的寄存器位宽有多大，在定长向量的CPU中，始终使用128bit，对于变长向量(比如SVE)，始终使用前128位

类型设计：
Batch<Tag> 表示SIMD数据类型，并且Batch类型是SIMD类型的直接别名，比如AVX2: using Batch<FullTag<double>> = __m256d
Mask<Tag>  表示SIMD掩码类型                                   AVX2: using Mask<FullTag<float>> = __m256;
                                                            AVX-512: using Mask<FullTag<float>> = __mmask16;

由于需要在 simd 256 的头文件中引入 fixed128 向量，所以对每个op的类型定义作出以下的统一规定：
定义Batch类型:
namespace detail
{
    template<typename Tag, typename Enable>
    struct batch_type;

    template<typename Tag>
    struct batch_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_32bits<Tag>>> // f32 256bits
    {
        using type = __m256;
    };
}

定义Mask类型:
namespace detail
{
    template<typename Tag, typename Enable> // Enable 不能有默认参数，因为有了默认参数，就不能重复定义了，到时候包含SSE的头文件的时候，就会标记重复定义
    struct mask_type;

    template<typename Tag>
    struct mask_type<Tag, std::enable_if_t<is_tag_256<Tag> && is_tag_float_32bits<Tag>>> // f32 256bits mask
    {
        using type = __m256;
    };
}

定义标量 Mask 类型(movemask所返回的mask类型):
namespace detail
{
    template<typename Tag, typename Enable>
    struct mask_bitset_type;

    template<typename Tag>
    struct mask_bitset_type<Tag, std::enable_if_t<is_tag_256<Tag>>>
    {
        using type = int; // 根据指令集的返回值来决定，比如X86返回的是int
    };
}

// declare user types
template<is_tag Tag> // 一定要用 is_tag，因为vec256会包含vec128的头文件，vec128的头文件的batch_type是128tag的特化，所以这里不能限死256
using Batch = typename detail::batch_type<Tag, void>::type;

template<is_tag Tag>
using Mask = typename detail::mask_type<Tag, void>::type;

template<is_tag Tag>
using MaskBitset = typename detail::mask_bitset_type<Tag, void>::type;

*/

namespace ksimd
{
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

    template<is_scalar_type S, size_t _bytes>
    struct Tag_base
    {
        using scalar_type = S;
        static constexpr size_t bytes = _bytes;
    };

    template<typename T>
    concept is_tag = std::is_base_of_v<Tag_base<typename T::scalar_type, T::bytes>, T>;


    // --- tag simd byte size ---
    template<typename Tag, size_t bytes>
    concept is_tag_bytes = is_tag<Tag> && (Tag::bytes == bytes);

    template<typename Tag>
    concept is_tag_scalar128 = is_tag_bytes<Tag, vec_size::Scalar128>;

    template<typename Tag>
    concept is_tag_128 = is_tag_bytes<Tag, vec_size::Vec128>;

    template<typename Tag>
    concept is_tag_256 = is_tag_bytes<Tag, vec_size::Vec256>;

    template<typename Tag>
    concept is_tag_512 = is_tag_bytes<Tag, vec_size::Vec512>;

    template<typename Tag>
    concept is_tag_scalable_full = is_tag_bytes<Tag, vec_size::Scalable>;


    // --- tag scalar type ---
    template<is_tag Tag>
    using tag_scalar_t = typename Tag::scalar_type;

    // signed
    template<typename Tag>
    concept is_tag_signed = is_tag<Tag> && is_scalar_signed<tag_scalar_t<Tag>>;

    // floating point
    template<typename Tag>
    concept is_tag_floating_point = is_tag<Tag> && is_scalar_floating_point<tag_scalar_t<Tag>>;

    // f32
    template<typename Tag>
    concept is_tag_float_32bits = is_tag<Tag> && is_scalar_type_float_32bits<tag_scalar_t<Tag>>;

    // f64
    template<typename Tag>
    concept is_tag_float_64bits = is_tag<Tag> && is_scalar_type_float_64bits<tag_scalar_t<Tag>>;


    // --- tag scalar bit size ---
    template<typename Tag, size_t bytes>
    concept is_tag_scalar_bytes = is_tag<Tag> && (sizeof(tag_scalar_t<Tag>) == bytes);

    template<typename Tag>
    concept is_tag_scalar_8 = is_tag_scalar_bytes<Tag, 1>;

    template<typename Tag>
    concept is_tag_scalar_16 = is_tag_scalar_bytes<Tag, 2>;

    template<typename Tag>
    concept is_tag_scalar_32 = is_tag_scalar_bytes<Tag, 4>;

    template<typename Tag>
    concept is_tag_scalar_64 = is_tag_scalar_bytes<Tag, 8>;
}

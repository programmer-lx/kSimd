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

    enum class TagType
    {
        FullTag,
        HalfIoTag,
        Fixed128Tag
    };

    template<is_scalar_type S, TagType Type>
    struct Tag_base
    {
        using scalar_type = S;
        static constexpr TagType tag_type = Type;
    };

    // tags
    template<is_scalar_type S>
    struct FullTag : Tag_base<S, TagType::FullTag> {};

    template<is_scalar_type S>
    struct HalfIoTag : Tag_base<S, TagType::HalfIoTag>
    {
        static_assert(Tag_base<S, TagType::HalfIoTag>::tag_type != TagType::HalfIoTag, "TODO: HalfIoTag");
    };

    template<is_scalar_type S>
    struct Fixed128Tag : Tag_base<S, TagType::Fixed128Tag>
    {
        static_assert(Tag_base<S, TagType::Fixed128Tag>::tag_type != TagType::Fixed128Tag, "TODO: Fixed128Tag");
    };

    template<typename T>
    concept is_tag = requires
    {
        typename T::scalar_type;
        T::tag_type;
        requires std::is_base_of_v<Tag_base<typename T::scalar_type, T::tag_type>, T>;
    };

    template<is_tag Tag>
    using tag_scalar_t = typename Tag::scalar_type;

    template<typename Tag, TagType... Types>
    KSIMD_HEADER_GLOBAL_CONSTEXPR bool tag_type_includes = []()
    {
        if constexpr (is_tag<Tag>)
        {
            return ((Tag::tag_type == Types) || ...);
        }
        else
        {
            return false;
        }
    }();

    // full
    template<typename Tag>
    concept is_tag_full = is_tag<Tag> && tag_type_includes<Tag, TagType::FullTag>;

    // fixed128
    template<typename Tag>
    concept is_tag_fixed128 = is_tag<Tag> && tag_type_includes<Tag, TagType::Fixed128Tag>;

    // full + fixed128
    template<typename Tag>
    concept is_tag_full_or_fixed128 = is_tag<Tag> && tag_type_includes<Tag, TagType::FullTag, TagType::Fixed128Tag>;

    // signed tag
    template<typename Tag>
    concept is_tag_signed = is_tag<Tag> && is_scalar_signed<tag_scalar_t<Tag>>;

    // floating point tag
    template<typename Tag>
    concept is_tag_float_point = is_tag<Tag> && is_scalar_floating_point<tag_scalar_t<Tag>>;

    // f32 tag
    template<typename Tag>
    concept is_tag_float_32bits = is_tag<Tag> && is_scalar_type_float_32bits<tag_scalar_t<Tag>>;
}

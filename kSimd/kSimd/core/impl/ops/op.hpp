#pragma once

#include "kSimd/core/impl/types.hpp"

namespace ksimd
{
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

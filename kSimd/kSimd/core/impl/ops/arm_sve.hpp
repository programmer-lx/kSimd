// do not use include guard

#include <arm_sve.h>

#include "kSimd/core/impl/dispatch.hpp"
#include "kSimd/core/impl/types.hpp"
#include "kSimd/core/impl/number.hpp"

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN static __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_scalable_full Tag>
    const size_t lanes(Tag) noexcept
    {
        constexpr size_t len = sizeof(tag_scalar_t<Tag>);

        static_assert(len == 1 || len == 2 || len == 4 || len == 8, "sizeof(scalar type) can only equal to 1, 2, 4, 8");

                if constexpr (len == 1)    return static_cast<size_t>(svcntb());
        else    if constexpr (len == 2)    return static_cast<size_t>(svcnth());
        else    if constexpr (len == 4)    return static_cast<size_t>(svcntw());
        else /* if constexpr (len == 8) */ return static_cast<size_t>(svcntd());
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Vec512;
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<typename Tag, typename Enable>
        struct batch_type;

        // f32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_float_32bits<Tag>>>
        {
            using type = svfloat32_t;
        };

        // f64
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_float_64bits<Tag>>>
        {
            using type = svfloat64_t;
        };
    } // namespace detail

    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = svbool_t;
#pragma endregion
}

#undef KSIMD_API

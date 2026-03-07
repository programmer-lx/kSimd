// do not use include guard

#include <arm_sve.h>

#include "shared.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

// 复用 NEON 的逻辑，实现 Fixed128Tag
#include "arm_neon.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#pragma region--- constants ---
    template<is_tag_scalable_full Tag>
    KSIMD_API(size_t) lanes(Tag) noexcept
    {
        // fake fp16 (promote to f32)
        #if KSIMD_DYN_DISPATCH_LEVEL != KSIMD_DYN_DISPATCH_LEVEL_SVE_FULL_FP16
        if constexpr (is_tag_float_16bits<Tag>)
        {
            return static_cast<size_t>(svcntw());
        }
        #endif

        constexpr size_t len = sizeof(tag_scalar_t<Tag>);

        static_assert(len == 1 || len == 2 || len == 4 || len == 8, "sizeof(scalar type) can only equal to 1, 2, 4, 8");

        if constexpr (len == 1) return static_cast<size_t>(svcntb());
        if constexpr (len == 2) return static_cast<size_t>(svcnth());
        if constexpr (len == 4) return static_cast<size_t>(svcntw());
        return static_cast<size_t>(svcntd()); /* if constexpr (len == 8) */
    }

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

        // int32
        template<typename Tag>
        struct batch_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag> && is_tag_int32<Tag>>>
        {
            using type = svint32_t;
        };

        template<typename Tag, typename Enable>
        struct mask_type;

        template<typename Tag>
        struct mask_type<Tag, std::enable_if_t<is_tag_scalable_full<Tag>>>
        {
            using type = svbool_t;
        };
    } // namespace detail

    template<is_tag Tag>
    using Batch = typename detail::batch_type<Tag, void>::type;

    template<is_tag Tag>
    using Mask = typename detail::mask_type<Tag, void>::type;
#pragma endregion
}

#undef KSIMD_API

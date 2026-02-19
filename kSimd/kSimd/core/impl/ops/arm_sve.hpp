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
#pragma region--- traits ---
    template<is_scalar_type S>
    struct Traits
    {
        using _scalar_type = S;
    };

    template<is_scalar_type S>
    const size_t lanes(Traits<S>) noexcept
    {
        constexpr size_t len = sizeof(S);

        static_assert(len == 1 || len == 2 || len == 4 || len == 8, "sizeof(S) can only equal to 1, 2, 4, 8");

               if constexpr (len == 1)   return svcntb();
        else   if constexpr (len == 2)   return svcnth();
        else   if constexpr (len == 4)   return svcntw();
        else /*if constexpr (len == 8)*/ return svcntd();
    }

    KSIMD_HEADER_GLOBAL_CONSTEXPR size_t Alignment = alignment::Vec512;
#pragma endregion

#pragma region--- types ---
    namespace detail
    {
        template<is_scalar_type>
        struct batch_type;

        template<>
        struct batch_type<float>
        {
            using type = svfloat32_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT32
        template<>
        struct batch_type<std::float32_t>
        {
            using type = svfloat32_t;
        };
#endif

        template<>
        struct batch_type<double>
        {
            using type = svfloat64_t;
        };

#if KSIMD_SUPPORT_STD_FLOAT64
        template<>
        struct batch_type<std::float64_t>
        {
            using type = svfloat64_t;
        };
#endif
    } // namespace detail

    template<is_scalar_type S>
    using Batch = typename detail::batch_type<S>::type;

    template<is_scalar_type S>
    using Mask = svbool_t;
#pragma endregion
}

#undef KSIMD_API

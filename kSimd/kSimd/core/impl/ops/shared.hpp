// do not use include guard

#include "op.hpp"
#include "kSimd/IDE/IDE_hint.hpp"

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
    using ksimd::RoundingMode;
    using ksimd::FloatMinMaxOption;

    namespace detail
    {
        template<is_scalar_type S>
        struct Tag128 : Tag_base<S , vec_size::Vec128> {};

        template<is_scalar_type S>
        struct Tag256 : Tag_base<S , vec_size::Vec256> {};

        template<is_scalar_type S>
        struct Tag512 : Tag_base<S , vec_size::Vec512> {};

        template<is_scalar_type S>
        struct TagScalable : Tag_base<S , vec_size::Scalable> {};
    }

    // tags
    template<is_scalar_type S>
    struct FullTag : Tag_base<S, ksimd::detail::dyn_vec_size::KSIMD_DYN_INSTRUCTION> {};

    template<is_scalar_type S>
    struct HalfIoTag : Tag_base<S, vec_size::Invalid> {};

    template<is_scalar_type S>
    struct Fixed128Tag : Tag_base<S , vec_size::Vec128> {};
}

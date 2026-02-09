// dot not use include guard

#include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd
{
    namespace KSIMD_DYN_INSTRUCTION
    {
        // ----------------- 二元算术运算 -----------------
        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator+(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::add(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator-(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::sub(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator*(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::mul(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator/(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::div(lhs, rhs);
        }

        // ----------------- 复合赋值算术运算 (Concise Style) -----------------
        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator+=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::add(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator-=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::sub(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator*=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::mul(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator/=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::div(lhs, rhs);
        }

        // ----------------- 位运算 (Bitwise) -----------------
        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator&(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::bit_and(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator|(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::bit_or(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator^(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::bit_xor(lhs, rhs);
        }

        // ----------------- 复合赋值位运算 (Concise Style) -----------------
        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator&=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::bit_and(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator|=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::bit_or(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Batch<S>&) operator^=(Batch<S>& lhs, Batch<S> rhs) noexcept
        {
            return lhs = op<S>::bit_xor(lhs, rhs);
        }

        // ----------------- 一元运算符 -----------------
        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator-(Batch<S> val) noexcept
        {
            return op<S>::neg(val);
        }
        template<is_scalar_type S>
        KSIMD_API(Batch<S>) operator~(Batch<S> val) noexcept
        {
            return op<S>::bit_not(val);
        }

        // ----------------- 比较 -----------------
        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator==(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::equal(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator!=(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::not_equal(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator>(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::greater(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator>=(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::greater_equal(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator<(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::less(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator<=(Batch<S> lhs, Batch<S> rhs) noexcept
        {
            return op<S>::less_equal(lhs, rhs);
        }

        // ----------------- Mask 逻辑运算符 (用于组合条件) -----------------
        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator&(Mask<S> lhs, Mask<S> rhs) noexcept
        {
            return op<S>::mask_and(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator|(Mask<S> lhs, Mask<S> rhs) noexcept
        {
            return op<S>::mask_or(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator^(Mask<S> lhs, Mask<S> rhs) noexcept
        {
            return op<S>::mask_xor(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>) operator~(Mask<S> mask) noexcept
        {
            return op<S>::mask_not(mask);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>&) operator&=(Mask<S>& lhs, Mask<S> rhs) noexcept
        {
            return lhs = op<S>::mask_and(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>&) operator|=(Mask<S>& lhs, Mask<S> rhs) noexcept
        {
            return lhs = op<S>::mask_or(lhs, rhs);
        }

        template<is_scalar_type S>
        KSIMD_API(Mask<S>&) operator^=(Mask<S>& lhs, Mask<S> rhs) noexcept
        {
            return lhs = op<S>::mask_xor(lhs, rhs);
        }
    } // namespace KSIMD_DYN_INSTRUCTION
} // namespace ksimd
#undef KSIMD_API
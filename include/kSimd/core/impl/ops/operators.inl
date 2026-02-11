// do not use include guard

// #include "kSimd/IDE/IDE_hint.hpp"

#define KSIMD_API(...) KSIMD_DYN_FUNC_ATTR KSIMD_FORCE_INLINE KSIMD_FLATTEN __VA_ARGS__ KSIMD_CALL_CONV

namespace ksimd::KSIMD_DYN_INSTRUCTION
{
#define KSIMD_BINARY_OP(symbol, name, ...) \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>) operator symbol(Batch<S> lhs, Batch<S> rhs) noexcept \
    { \
        __VA_ARGS__ \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, rhs); \
    } \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>&) operator symbol## = (Batch<S> & lhs, Batch<S> rhs) noexcept \
    { \
        __VA_ARGS__ \
        return lhs = ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, rhs); \
    }

    // ----------------- 二元算术运算 -----------------
    KSIMD_BINARY_OP(+, add)
    KSIMD_BINARY_OP(-, sub)
    KSIMD_BINARY_OP(*, mul)
    KSIMD_BINARY_OP(/, div,
                    static_assert(is_scalar_floating_point<S>, "operator/ can only be used by floating point.");)

    // ----------------- 二元位运算 -----------------
    KSIMD_BINARY_OP(&, bit_and)
    KSIMD_BINARY_OP(|, bit_or)
    KSIMD_BINARY_OP(^, bit_xor)

#undef KSIMD_BINARY_OP

#define KSIMD_UNARY_OP(symbol, name) \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>) operator symbol(Batch<S> val) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(val); \
    }

    // -----------------一元算术运算 -----------------
    KSIMD_UNARY_OP(-, neg)
    // ----------------- 一元位运算 -----------------
    KSIMD_UNARY_OP(~, bit_not)

#undef KSIMD_UNARY_OP

#define KSIMD_MIXED_BINARY_OP(symbol, name, ...) \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>) operator symbol(Batch<S> lhs, S rhs) noexcept \
    { \
        __VA_ARGS__ \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, set(rhs)); \
    } \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>) operator symbol(S lhs, Batch<S> rhs) noexcept \
    { \
        __VA_ARGS__ \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(set(lhs), rhs); \
    } \
    template<is_scalar_type S> \
    KSIMD_API(Batch<S>&) operator symbol## = (Batch<S> & lhs, S rhs) noexcept \
    { \
        __VA_ARGS__ \
        return lhs = ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, set(rhs)); \
    }

    // ----------------- 与标量混合的二元算术运算符 -----------------
    KSIMD_MIXED_BINARY_OP(+, add)
    KSIMD_MIXED_BINARY_OP(-, sub)
    KSIMD_MIXED_BINARY_OP(*, mul)
    KSIMD_MIXED_BINARY_OP(/, div,
                          static_assert(is_scalar_floating_point<S>, "operator/ can only be used by floating point.");)

    // ----------------- 与标量混合的二元位运算符 -----------------
    KSIMD_MIXED_BINARY_OP(&, bit_and)
    KSIMD_MIXED_BINARY_OP(|, bit_or)
    KSIMD_MIXED_BINARY_OP(^, bit_xor)

#undef KSIMD_MIXED_BINARY_OP

// ----------------- 比较 -----------------
#define KSIMD_COMP_OP(symbol, name) \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>) operator symbol(Batch<S> lhs, Batch<S> rhs) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, rhs); \
    }

    KSIMD_COMP_OP(==, equal)
    KSIMD_COMP_OP(!=, not_equal)
    KSIMD_COMP_OP(<, less)
    KSIMD_COMP_OP(<=, less_equal)
    KSIMD_COMP_OP(>, greater)
    KSIMD_COMP_OP(>=, greater_equal)

#undef KSIMD_COMP_OP

// ----------------- 与标量混合的比较运算符 -----------------
#define KSIMD_MIXED_COMP_OP(symbol, name) \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>) operator symbol(Batch<S> lhs, S rhs) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, set(rhs)); \
    } \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>) operator symbol(S lhs, Batch<S> rhs) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(set(lhs), rhs); \
    }

    KSIMD_MIXED_COMP_OP(==, equal)
    KSIMD_MIXED_COMP_OP(!=, not_equal)
    KSIMD_MIXED_COMP_OP(>, greater)
    KSIMD_MIXED_COMP_OP(>=, greater_equal)
    KSIMD_MIXED_COMP_OP(<, less)
    KSIMD_MIXED_COMP_OP(<=, less_equal)

#undef KSIMD_MIXED_COMP_OP

// ----------------- Mask 二元逻辑运算符 (用于组合条件) -----------------
#define KSIMD_MASK_BINARY_LOGIC_OP(symbol, name) \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>) operator symbol(Mask<S> lhs, Mask<S> rhs) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, rhs); \
    } \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>&) operator symbol## = (Mask<S> & lhs, Mask<S> rhs) noexcept \
    { \
        return lhs = ksimd::KSIMD_DYN_INSTRUCTION::name(lhs, rhs); \
    }

    KSIMD_MASK_BINARY_LOGIC_OP(&, mask_and)
    KSIMD_MASK_BINARY_LOGIC_OP(|, mask_or)
    KSIMD_MASK_BINARY_LOGIC_OP(^, mask_xor)

#undef KSIMD_MASK_BINARY_LOGIC_OP

#define KSIMD_MASK_UNARY_LOGIC_OP(symbol, name) \
    template<is_scalar_type S> \
    KSIMD_API(Mask<S>) operator symbol(Mask<S> mask) noexcept \
    { \
        return ksimd::KSIMD_DYN_INSTRUCTION::name(mask); \
    }

    KSIMD_MASK_UNARY_LOGIC_OP(~, mask_not)
    // KSIMD_MASK_UNARY_LOGIC_OP(!, mask_not) // 不重载!取反，只使用~按位运算

#undef KSIMD_MASK_UNARY_LOGIC_OP
} // namespace ksimd::KSIMD_DYN_INSTRUCTION

#undef KSIMD_API

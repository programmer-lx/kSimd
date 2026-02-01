// using ALL_TYPE_T = uint32_t;

#include "../../test.hpp"

#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "simd_op/ALL_TYPE_T/all_type.inl" // this file
#include <kSimd/dispatch_this_file.hpp> // auto dispatch
#include <kSimd/simd_op.hpp>

using namespace ksimd;

// ------------------------------------------ zero ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void zero(TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);

        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::zero());
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(zero);

TEST(dyn_dispatch_TYPE_T, zero)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(zero)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T out[TOTAL];

        // 先填充非零值
        for (size_t i = 0; i < TOTAL; ++i) out[i] = static_cast<TYPE_T>(123);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(zero)[idx](out);

        // 检查结果是否全为 0
        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == static_cast<TYPE_T>(0));
    }
}
#endif

// ------------------------------------------ set ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void set(TYPE_T x, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::set(x));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(set);

TEST(dyn_dispatch_TYPE_T, set)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) out[i] = static_cast<TYPE_T>(123);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(set)[idx](static_cast<TYPE_T>(3), out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == static_cast<TYPE_T>(3));
    }
}
#endif

// ------------------------------------------ load + store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void load_store(const TYPE_T* KSIMD_RESTRICT in, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::store(out + i, op::load(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(load_store);

TEST(dyn_dispatch_TYPE_T, load_store)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = TYPE_T(i + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = TYPE_T(456);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(load_store)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == in[i]);
    }
}
#endif

// ------------------------------------------ loadu + storeu ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void loadu_storeu(const TYPE_T* KSIMD_RESTRICT in, TYPE_T* KSIMD_RESTRICT out) noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        constexpr size_t Step = op::Lanes;

        for (size_t i = 0; i < TOTAL; i += Step)
        {
            op::storeu(out + i, op::loadu(in + i));
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(loadu_storeu);

TEST(dyn_dispatch_TYPE_T, loadu_storeu)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)); ++idx)
    {
        alignas(ALIGNMENT) TYPE_T in[TOTAL], out[TOTAL];

        for (size_t i = 0; i < TOTAL; ++i) in[i] = TYPE_T(i * 2 + 1);
        for (size_t i = 0; i < TOTAL; ++i) out[i] = TYPE_T(999);

        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(loadu_storeu)[idx](in, out);

        for (size_t i = 0; i < TOTAL; ++i)
            EXPECT_TRUE(out[i] == in[i]);
    }
}
#endif

// ------------------------------------------ mask_load mask_store ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_load_mask_store() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        using mask_t = op::mask_t;
        using batch_t = op::batch_t;
        constexpr size_t lanes = op::Lanes;

        alignas(ALIGNMENT) TYPE_T arr[lanes] = {};
        FILL_ARRAY(arr, TYPE_T(99));

        {
            mask_t mask = op::mask_from_lanes(lanes - 1);
            batch_t data = op::mask_load(arr, mask);
            alignas(ALIGNMENT) TYPE_T result[lanes]{};
            op::store(result, data);
            for (size_t i = 0; i < lanes; ++i)
            {
                if (i < lanes - 1)
                {
                    EXPECT_TRUE(result[i] == TYPE_T(99));
                }
                else
                {
                    EXPECT_TRUE(result[i] == TYPE_T(0));
                }
            }
        }
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(mask_load_mask_store);

TEST(dyn_dispatch_TYPE_T, mask_load_mask_store)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_load_mask_store)); ++idx)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_load_mask_store)[idx]();
    }
}
#endif

// ------------------------------------------ mask_from_lanes ------------------------------------------
namespace KSIMD_DYN_INSTRUCTION
{
    KSIMD_DYN_FUNC_ATTR
    void mask_from_lanes() noexcept
    {
        using op = KSIMD_DYN_SIMD_OP(TYPE_T);
        using mask_t = op::mask_t;
        constexpr size_t lanes = op::Lanes;

        alignas(ALIGNMENT) TYPE_T arr[lanes] = {};

        mask_t mask = op::mask_from_lanes(lanes - 1);
        FILL_ARRAY(arr, one_block<TYPE_T>);
        arr[lanes - 1] = TYPE_T(0);
        EXPECT_TRUE(simd_type_bit_equal(mask, arr));

        mask = op::mask_from_lanes(1000);
        FILL_ARRAY(arr, one_block<TYPE_T>);
        EXPECT_TRUE(simd_type_bit_equal(mask, arr));

        mask = op::mask_from_lanes(0);
        FILL_ARRAY(arr, zero_block<TYPE_T>);
        EXPECT_TRUE(simd_type_bit_equal(mask, arr));
    }
}

#if KSIMD_ONCE
KSIMD_DYN_DISPATCH_FUNC(mask_from_lanes);

TEST(dyn_dispatch_TYPE_T, mask_from_lanes)
{
    for (size_t idx = 0; idx < std::size(KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_from_lanes)); ++idx)
    {
        KSIMD_DETAIL_PFN_TABLE_FULL_NAME(mask_from_lanes)[idx]();
    }
}
#endif


#if KSIMD_ONCE
int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
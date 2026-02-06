#undef KSIMD_DISPATCH_THIS_FILE
#define KSIMD_DISPATCH_THIS_FILE "multi_simd_var.cpp"
#include <kSimd/dispatch_this_file.hpp>

#include <kSimd/base_op.hpp>
#include <kSimd/fixed_op.hpp>

namespace KSIMD_DYN_INSTRUCTION
{
    // 定义 batch_t 为 2 个 __m128，总宽 8 个 float
    using f4x2 = KSIMD_DYN_FIXED_OP(float, 4, 2);

    /*
     * 测试场景：条件混合运算
     * 逻辑：
     * 1. 比较 a 和 b
     * 2. 如果 a > b，计算 result = -(a * b + c)
     * 3. 否则，       计算 result = -(a - b)
     * * 测试点：
     * - 寄存器分配：涉及 a, b, c, mask, v_true, v_false, result 多个变量，看是否溢出到栈。
     * - 指令融合：mask_select 展开后的 and/andnot/or 是否高效。
     * - 冗余消除：最后的 neg 操作是否能正确应用。
     * - 尾部处理：复杂的 mask_load/store 是否生成正确的标量跳转表。
     */
    KSIMD_DYN_FUNC_ATTR void complex_kernel(
        const float* KSIMD_RESTRICT a,
        const float* KSIMD_RESTRICT b,
        const float* KSIMD_RESTRICT c,
              float* KSIMD_RESTRICT out,
        const size_t                size
    ) noexcept
    {
        size_t i = 0;

        // --- 主循环 (Loop Body) ---
        // 编译器应将其展开为两组独立的 SSE 指令流，并进行指令交织
        for (; i + f4x2::TotalLanes <= size; i += f4x2::TotalLanes)
        {
            // 1. Load (占用 6 个 xmm 寄存器: a0,a1, b0,b1, c0,c1)
            f4x2::batch_t va = f4x2::load(a + i);
            f4x2::batch_t vb = f4x2::load(b + i);
            f4x2::batch_t vc = f4x2::load(c + i);

            // 2. Compare (生成 mask0, mask1)
            f4x2::mask_t mask = f4x2::greater(va, vb);

            // 3. True Branch: Mul+Add
            f4x2::batch_t v_true = f4x2::mul_add(va, vb, vc);

            // 4. False Branch: Sub
            f4x2::batch_t v_false = f4x2::sub(va, vb);

            // 5. Select (Blend)
            // 展开为: (mask & v_true) | (~mask & v_false)
            f4x2::batch_t res = f4x2::mask_select(mask, v_true, v_false);

            // 6. Bitwise Op (Negate)
            // 测试编译器能否看穿数据流
            res = f4x2::neg(res);

            // 7. Store
            f4x2::store(out + i, res);
        }

        // --- 尾部处理 (Tail Handling) ---
        // 测试 Mask Load/Store 的实现质量
        if (const size_t rest = size - i; rest > 0)
        {
            f4x2::mask_t mask = f4x2::mask_from_lanes(rest);

            // 安全加载 (即使越界也只读取有效部分)
            f4x2::batch_t va = f4x2::mask_load(a + i, mask);
            f4x2::batch_t vb = f4x2::mask_load(b + i, mask);
            f4x2::batch_t vc = f4x2::mask_load(c + i, mask);

            f4x2::mask_t cmp_mask = f4x2::greater(va, vb);

            f4x2::batch_t v_true = f4x2::mul_add(va, vb, vc);
            f4x2::batch_t v_false = f4x2::sub(va, vb);

            // 这里的 mask_select 使用的是比较结果 cmp_mask
            f4x2::batch_t res = f4x2::mask_select(cmp_mask, v_true, v_false);

            res = f4x2::neg(res);

            // 安全存储
            f4x2::mask_store(out + i, res, mask);
        }
    }
}
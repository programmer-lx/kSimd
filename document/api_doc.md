# API 文档

---

## 1) `load`
**签名**：`template<typename Tag> Batch<Tag> load(Tag tag, const tag_scalar_t<Tag>* mem) noexcept`  
**参数**：`tag: Tag`，`mem: const tag_scalar_t<Tag>*`  
**返回类型**：`Batch<Tag>`  
**返回语义**：从 `mem` 读取一个完整向量并返回。

## 2) `store`
**签名**：`template<typename Tag> void store(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`  
**返回类型**：`void`  
**返回语义**：将 `v` 全部 lane 写入 `mem`。

## 3) `loadu`
**签名**：`template<typename Tag> Batch<Tag> loadu(Tag tag, const tag_scalar_t<Tag>* mem) noexcept`  
**参数**：`tag: Tag`，`mem: const tag_scalar_t<Tag>*`  
**返回类型**：`Batch<Tag>`  
**返回语义**：从非对齐地址 `mem` 读取一个完整向量。

## 4) `storeu`
**签名**：`template<typename Tag> void storeu(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`  
**返回类型**：`void`  
**返回语义**：将 `v` 非对齐写入 `mem`。

## 5) `loadu_partial`
**签名**：`template<typename Tag> Batch<Tag> loadu_partial(Tag tag, const tag_scalar_t<Tag>* mem, size_t count) noexcept`  
**参数**：`tag: Tag`，`mem: const tag_scalar_t<Tag>*`，`count: size_t`  
**返回类型**：`Batch<Tag>`  
**返回语义**：加载前 `count` 个元素，其余 lane 置 0。

## 6) `storeu_partial`
**签名**：`template<typename Tag> void storeu_partial(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count) noexcept`  
**参数**：`tag: Tag`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`，`count: size_t`  
**返回类型**：`void`  
**返回语义**：将 `v` 的前 `count` 个元素写入 `mem`。

## 7) `undefined`
**签名**：`template<typename Tag> Batch<Tag> undefined(Tag tag) noexcept`  
**参数**：`tag: Tag`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回未定义语义的向量值（按类型分支）。

## 8) `zero`
**签名**：`template<typename Tag> Batch<Tag> zero(Tag tag) noexcept`  
**参数**：`tag: Tag`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回全 0 向量。

## 9) `set`
**签名**：`template<typename Tag> Batch<Tag> set(Tag tag, tag_scalar_t<Tag> x) noexcept`  
**参数**：`tag: Tag`，`x: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回所有 lane 都为 `x` 的向量。

## 10) `sequence(Tag)`
**签名**：`template<typename Tag> Batch<Tag> sequence(Tag tag) noexcept`  
**参数**：`tag: Tag`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回从 0 开始、步长为 1 的递增序列向量。

## 11) `sequence(Tag, base)`
**签名**：`template<typename Tag> Batch<Tag> sequence(Tag tag, tag_scalar_t<Tag> base) noexcept`  
**参数**：`tag: Tag`，`base: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回从 `base` 开始、步长为 1 的序列向量。

## 12) `sequence(Tag, base, stride)`
**签名**：`template<typename Tag> Batch<Tag> sequence(Tag tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride) noexcept`  
**参数**：`tag: Tag`，`base: tag_scalar_t<Tag>`，`stride: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回 `base + i * stride`（`i` 为 lane 下标）的序列向量。

## 13) `add`
**签名**：`template<typename Tag> Batch<Tag> add(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 加法结果。

## 14) `sub`
**签名**：`template<typename Tag> Batch<Tag> sub(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 减法 `lhs-rhs` 结果。

## 15) `mul`
**签名**：`template<typename Tag> Batch<Tag> mul(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 乘法结果。

## 16) `mul_add`
**签名**：`template<typename Tag> Batch<Tag> mul_add(Tag tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c) noexcept`  
**参数**：`tag: Tag`，`a: Batch<Tag>`，`b: Batch<Tag>`，`c: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 计算 `a*b+c`。

## 17) `min`
**签名**：`template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag> Batch<Tag> min(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 最小值结果。

## 18) `max`
**签名**：`template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag> Batch<Tag> max(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 最大值结果。

## 19) `bit_not`
**签名**：`template<typename Tag> Batch<Tag> bit_not(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位取反。  
**备注**：`float16` 分支在该文件中为删除声明。

## 20) `bit_and`
**签名**：`template<typename Tag> Batch<Tag> bit_and(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位与。  
**备注**：`float16` 分支在该文件中为删除声明。

## 21) `bit_and_not`
**签名**：`template<typename Tag> Batch<Tag> bit_and_not(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane `~lhs & rhs`。  
**备注**：`float16` 分支在该文件中为删除声明。

## 22) `bit_or`
**签名**：`template<typename Tag> Batch<Tag> bit_or(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位或。  
**备注**：`float16` 分支在该文件中为删除声明。

## 23) `bit_xor`
**签名**：`template<typename Tag> Batch<Tag> bit_xor(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位异或。  
**备注**：`float16` 分支在该文件中为删除声明。

## 24) `equal`
**签名**：`template<typename Tag> Mask<Tag> equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs == rhs` 掩码。

## 25) `not_equal`
**签名**：`template<typename Tag> Mask<Tag> not_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs != rhs` 掩码。

## 26) `greater`
**签名**：`template<typename Tag> Mask<Tag> greater(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs > rhs` 掩码。

## 27) `greater_equal`
**签名**：`template<typename Tag> Mask<Tag> greater_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs >= rhs` 掩码。

## 28) `less`
**签名**：`template<typename Tag> Mask<Tag> less(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs < rhs` 掩码。

## 29) `less_equal`
**签名**：`template<typename Tag> Mask<Tag> less_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs <= rhs` 掩码。

## 30) `mask_and`
**签名**：`template<typename Tag> Mask<Tag> mask_and(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位与。

## 31) `mask_or`
**签名**：`template<typename Tag> Mask<Tag> mask_or(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位或。

## 32) `mask_xor`
**签名**：`template<typename Tag> Mask<Tag> mask_xor(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位异或。

## 33) `mask_not`
**签名**：`template<typename Tag> Mask<Tag> mask_not(Tag tag, Mask<Tag> mask) noexcept`  
**参数**：`tag: Tag`，`mask: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位取反。

## 34) `mask_and_not`
**签名**：`template<typename Tag> Mask<Tag> mask_and_not(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码 `~lhs & rhs`。

## 35) `mask_all`
**签名**：`template<typename Tag> bool mask_all(Tag tag, Mask<Tag> mask) noexcept`  
**参数**：`tag: Tag`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：所有 lane 是否都为真。

## 36) `mask_any`
**签名**：`template<typename Tag> bool mask_any(Tag tag, Mask<Tag> mask) noexcept`  
**参数**：`tag: Tag`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：是否存在任一 lane 为真。

## 37) `mask_none`
**签名**：`template<typename Tag> bool mask_none(Tag tag, Mask<Tag> mask) noexcept`  
**参数**：`tag: Tag`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：是否所有 lane 都为假。

## 38) `if_then_else`
**签名**：`template<typename Tag> Batch<Tag> if_then_else(Tag tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else) noexcept`  
**参数**：`tag: Tag`，`_if: Mask<Tag>`，`_then: Batch<Tag>`，`_else: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：按 `_if` 掩码逐 lane 选择 `_then` 或 `_else`。

## 39) `reduce_add`
**签名**：`template<typename Tag> /* tag_scalar_t<Tag> 或 int32_t */ reduce_add(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>` 或 `int32_t`（依类型分支）  
**返回语义**：将所有 lane 求和并返回标量。

## 40) `reduce_mul`
**签名**：`template<typename Tag> /* tag_scalar_t<Tag> 或 int32_t */ reduce_mul(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>` 或 `int32_t`（依类型分支）  
**返回语义**：将所有 lane 连乘并返回标量。

## 41) `reduce_min`
**签名**：`template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag> tag_scalar_t<Tag> reduce_min(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>`  
**返回语义**：返回 `v` 所有 lane 的最小值标量。

## 42) `reduce_max`
**签名**：`template<FloatMinMaxOption option = FloatMinMaxOption::Native, typename Tag> tag_scalar_t<Tag> reduce_max(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>`  
**返回语义**：返回 `v` 所有 lane 的最大值标量。

## 43) `abs`
**签名**：`template<typename Tag> Batch<Tag> abs(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 绝对值结果。

## 44) `neg`
**签名**：`template<typename Tag> Batch<Tag> neg(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 取负结果。

## 45) `div`
**签名**：`template<typename Tag> Batch<Tag> div(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 浮点除法 `lhs/rhs` 结果。

## 46) `sqrt`
**签名**：`template<typename Tag> Batch<Tag> sqrt(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 平方根结果。

## 47) `round`
**签名**：`template<RoundingMode mode, typename Tag> Batch<Tag> round(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`mode: RoundingMode`，`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：按 `mode` 对每个 lane 舍入后的向量。

## 48) `not_greater`
**签名**：`template<typename Tag> Mask<Tag> not_greater(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非大于”判断掩码。

## 49) `not_greater_equal`
**签名**：`template<typename Tag> Mask<Tag> not_greater_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非大于等于”判断掩码。

## 50) `not_less`
**签名**：`template<typename Tag> Mask<Tag> not_less(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非小于”判断掩码。

## 51) `not_less_equal`
**签名**：`template<typename Tag> Mask<Tag> not_less_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非小于等于”判断掩码。

## 52) `any_NaN`
**签名**：`template<typename Tag> Mask<Tag> any_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs`/`rhs` 任一为 NaN 的掩码。

## 53) `all_NaN`
**签名**：`template<typename Tag> Mask<Tag> all_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 与 `rhs` 均为 NaN 的掩码。

## 54) `not_NaN`
**签名**：`template<typename Tag> Mask<Tag> not_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断两者都不是 NaN 的掩码。

## 55) `any_finite`
**签名**：`template<typename Tag> Mask<Tag> any_finite(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 或 `rhs` 任一为有限值的掩码。

## 56) `all_finite`
**签名**：`template<typename Tag> Mask<Tag> all_finite(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs) noexcept`  
**参数**：`tag: Tag`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 与 `rhs` 均为有限值的掩码。

## 57) `rcp`
**签名**：`template<typename Tag> Batch<Tag> rcp(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 倒数近似值向量。

## 58) `rsqrt`
**签名**：`template<typename Tag> Batch<Tag> rsqrt(Tag tag, Batch<Tag> v) noexcept`  
**参数**：`tag: Tag`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 逆平方根近似值向量。

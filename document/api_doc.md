# API 文档

---

## `load`
**签名**：`load(Tag tag, const tag_scalar_t<Tag>* mem)`  
**参数**：`tag: Tag (any)`，`mem: const tag_scalar_t<Tag>*`  
**返回类型**：`Batch<Tag>`  
**返回语义**：从 `mem` 读取一个完整向量并返回。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Performance) float in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<float>(i + 1);
// 示例代码:
auto v = ns::load(tag, in_data);
// 效果:
// 从 `mem` 读取一个完整向量并返回。
```

## `store`
**签名**：`store(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`  
**返回类型**：`void`  
**返回语义**：将 `v` 全部 lane 写入 `mem`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
auto v = ns::set(tag, 2.0);
alignas(ksimd::alignment::Required) double out_data[L] = {};
// 示例代码:
ns::store(tag, out_data, v);
// 效果:
// 将 `v` 全部 lane 写入 `mem`。
```

## `loadu`
**签名**：`loadu(Tag tag, const tag_scalar_t<Tag>* mem)`  
**参数**：`tag: Tag (any)`，`mem: const tag_scalar_t<Tag>*`  
**返回类型**：`Batch<Tag>`  
**返回语义**：从非对齐地址 `mem` 读取一个完整向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t raw[L + 1] = {};
int32_t* in_data = raw + 1; // 故意非对齐
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<int32_t>(i + 10);
// 示例代码:
auto v = ns::loadu(tag, in_data);
// 效果:
// 从非对齐地址 `mem` 读取一个完整向量。
```

## `storeu`
**签名**：`storeu(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`  
**返回类型**：`void`  
**返回语义**：将 `v` 非对齐写入 `mem`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
auto v = ns::set(tag, static_cast<uint8_t>(7));
uint8_t out_data[L + 1] = {};
uint8_t* p = out_data + 1; // 非对齐地址
// 示例代码:
ns::storeu(tag, p, v);
// 效果:
// 将 `v` 非对齐写入 `mem`。
```

## `loadu_partial`
**签名**：`loadu_partial(Tag tag, const tag_scalar_t<Tag>* mem, size_t count)`  
**参数**：`tag: Tag (any)`，`mem: const tag_scalar_t<Tag>*`，`count: size_t`  
**返回类型**：`Batch<Tag>`  
**返回语义**：加载前 `count` 个元素，其余 lane 置 0。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
float in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<float>(i + 1);
size_t count = L / 2;
// 示例代码:
auto v = ns::loadu_partial(tag, in_data, count);
// 效果:
// 加载前 `count` 个元素，其余 lane 置 0。
```

## `storeu_partial`
**签名**：`storeu_partial(Tag tag, tag_scalar_t<Tag>* mem, Batch<Tag> v, size_t count)`  
**参数**：`tag: Tag (any)`，`mem: tag_scalar_t<Tag>*`，`v: Batch<Tag>`，`count: size_t`  
**返回类型**：`void`  
**返回语义**：将 `v` 的前 `count` 个元素写入 `mem`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
auto v = ns::sequence(tag);
int32_t out_data[L] = {};
size_t count = L / 2;
// 示例代码:
ns::storeu_partial(tag, out_data, v, count);
// 效果:
// 将 `v` 的前 `count` 个元素写入 `mem`。
```

## `undefined`
**签名**：`undefined(Tag tag)`  
**参数**：`tag: Tag (any)`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回未定义语义的向量值（按类型分支）。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
// 示例代码:
auto v = ns::undefined(tag);
// 效果:
// 返回未定义语义的向量值（按类型分支）。
```

## `zero`
**签名**：`zero(Tag tag)`  
**参数**：`tag: Tag (any)`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回全 0 向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
// 示例代码:
auto v = ns::zero(tag);
// 效果:
// 返回全 0 向量。
```

## `set`
**签名**：`set(Tag tag, tag_scalar_t<Tag> x)`  
**参数**：`tag: Tag (any)`，`x: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回所有 lane 都为 `x` 的向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t x = 9;
// 示例代码:
auto v = ns::set(tag, x);
// 效果:
// 返回所有 lane 都为 `x` 的向量。
```

## `sequence(Tag)`
**签名**：`sequence(Tag tag)`  
**参数**：`tag: Tag (any)`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回从 0 开始、步长为 1 的递增序列向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
// 示例代码:
auto v = ns::sequence(tag);
// 效果:
// 返回从 0 开始、步长为 1 的递增序列向量。
```

## `sequence(Tag, base)`
**签名**：`sequence(Tag tag, tag_scalar_t<Tag> base)`  
**参数**：`tag: Tag (any)`，`base: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回从 `base` 开始、步长为 1 的序列向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
double base = 10.0;
// 示例代码:
auto v = ns::sequence(tag, base);
// 效果:
// 返回从 `base` 开始、步长为 1 的序列向量。
```

## `sequence(Tag, base, stride)`
**签名**：`sequence(Tag tag, tag_scalar_t<Tag> base, tag_scalar_t<Tag> stride)`  
**参数**：`tag: Tag (any)`，`base: tag_scalar_t<Tag>`，`stride: tag_scalar_t<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：返回 `base + i * stride`（`i` 为 lane 下标）的序列向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t base = 5;
int32_t stride = 3;
// 示例代码:
auto v = ns::sequence(tag, base, stride);
// 效果:
// 返回 `base + i * stride`（`i` 为 lane 下标）的序列向量。
```

## `add`
**签名**：`add(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 加法结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) uint8_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) uint8_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<uint8_t>((i + 1) & 0xFF); rhs_data[i] = static_cast<uint8_t>((i + 3) & 0xFF); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::add(tag, lhs, rhs);
// 效果:
// 逐 lane 加法结果。
```

## `sub`
**签名**：`sub(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 减法 `lhs-rhs` 结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 1); rhs_data[i] = static_cast<float>(i + 3); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::sub(tag, lhs, rhs);
// 效果:
// 逐 lane 减法 `lhs-rhs` 结果。
```

## `mul`
**签名**：`mul(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 乘法结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 1); rhs_data[i] = static_cast<double>(i + 3); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::mul(tag, lhs, rhs);
// 效果:
// 逐 lane 乘法结果。
```

## `mul_add`
**签名**：`mul_add(Tag tag, Batch<Tag> a, Batch<Tag> b, Batch<Tag> c)`  
**参数**：`tag: Tag (any)`，`a: Batch<Tag>`，`b: Batch<Tag>`，`c: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 计算 `a*b+c`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t a_data[L] = {}; int32_t b_data[L] = {}; int32_t c_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<int32_t>(i + 1); b_data[i] = 2; c_data[i] = 1; }
// 示例代码:
auto a = ns::loadu(tag, a_data);
auto b = ns::loadu(tag, b_data);
auto c = ns::loadu(tag, c_data);
auto result = ns::mul_add(tag, a, b, c);
// 效果:
// 逐 lane 计算 `a*b+c`。
```

## `min`
**签名**：`template<FloatMinMaxOption = Native> min(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 最小值结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
float a_data[L] = {}; float b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<float>(i + 1); b_data[i] = static_cast<float>(L - i); }
// 示例代码:
auto a = ns::loadu(tag, a_data);
auto b = ns::loadu(tag, b_data);
auto result = ns::min<ns::FloatMinMaxOption::CheckNaN>(tag, a, b);
// 效果:
// 逐 lane 最小值结果。
```

## `max`
**签名**：`template<FloatMinMaxOption = Native> max(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 最大值结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
double a_data[L] = {}; double b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<double>(i + 1); b_data[i] = static_cast<double>(L - i); }
// 示例代码:
auto a = ns::loadu(tag, a_data);
auto b = ns::loadu(tag, b_data);
auto result = ns::max<ns::FloatMinMaxOption::Native>(tag, a, b);
// 效果:
// 逐 lane 最大值结果。
```

## `bit_not`
**签名**：`bit_not(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位取反。  
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<int32_t>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::bit_not(tag, v);
// 效果:
// 逐 lane 按位取反。
```
**备注**：`float16` 分支不可用。

## `bit_and`
**签名**：`bit_and(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位与。  
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) uint8_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) uint8_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<uint8_t>((i + 1) & 0xFF); rhs_data[i] = static_cast<uint8_t>((i + 3) & 0xFF); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::bit_and(tag, lhs, rhs);
// 效果:
// 逐 lane 按位与。
```
**备注**：`float16` 分支不可用。

## `bit_and_not`
**签名**：`bit_and_not(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane `~lhs & rhs`。  
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) int32_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<int32_t>(i + 1); rhs_data[i] = static_cast<int32_t>(i + 3); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::bit_and_not(tag, lhs, rhs);
// 效果:
// 逐 lane `~lhs & rhs`。
```
**备注**：`float16` 分支不可用。

## `bit_or`
**签名**：`bit_or(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位或。  
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) uint8_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) uint8_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<uint8_t>((i + 1) & 0xFF); rhs_data[i] = static_cast<uint8_t>((i + 3) & 0xFF); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::bit_or(tag, lhs, rhs);
// 效果:
// 逐 lane 按位或。
```
**备注**：`float16` 分支不可用。

## `bit_xor`
**签名**：`bit_xor(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 按位异或。  
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) int32_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<int32_t>(i + 1); rhs_data[i] = static_cast<int32_t>(i + 3); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::bit_xor(tag, lhs, rhs);
// 效果:
// 逐 lane 按位异或。
```
**备注**：`float16` 分支不可用。

## `equal`
**签名**：`equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs == rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::equal(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs == rhs` 掩码。
```

## `not_equal`
**签名**：`not_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs != rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 2); rhs_data[i] = static_cast<double>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_equal(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs != rhs` 掩码。
```

## `greater`
**签名**：`greater(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs > rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) int32_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<int32_t>(i + 2); rhs_data[i] = static_cast<int32_t>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::greater(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs > rhs` 掩码。
```

## `greater_equal`
**签名**：`greater_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs >= rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) uint8_t lhs_data[L] = {};
alignas(ksimd::alignment::Required) uint8_t rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<uint8_t>((i + 2) & 0xFF); rhs_data[i] = static_cast<uint8_t>((i + 1) & 0xFF); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::greater_equal(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs >= rhs` 掩码。
```

## `less`
**签名**：`less(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs < rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::less(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs < rhs` 掩码。
```

## `less_equal`
**签名**：`less_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane `lhs <= rhs` 掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 1); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::less_equal(tag, lhs, rhs);
// 效果:
// 逐 lane `lhs <= rhs` 掩码。
```

## `mask_and`
**签名**：`mask_and(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位与。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t a_data[L] = {};
alignas(ksimd::alignment::Required) int32_t b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<int32_t>(i + 2); b_data[i] = static_cast<int32_t>(i + 1); }
auto va = ns::loadu(tag, a_data);
auto vb = ns::loadu(tag, b_data);
auto ma = ns::greater(tag, va, vb);
auto mb = ns::less_equal(tag, va, vb);
// 示例代码:
auto mask = ns::mask_and(tag, ma, mb);
// 效果:
// 掩码按位与。
```

## `mask_or`
**签名**：`mask_or(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位或。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) uint8_t a_data[L] = {};
alignas(ksimd::alignment::Required) uint8_t b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<uint8_t>((i + 2) & 0xFF); b_data[i] = static_cast<uint8_t>((i + 1) & 0xFF); }
auto va = ns::loadu(tag, a_data);
auto vb = ns::loadu(tag, b_data);
auto ma = ns::greater(tag, va, vb);
auto mb = ns::less_equal(tag, va, vb);
// 示例代码:
auto mask = ns::mask_or(tag, ma, mb);
// 效果:
// 掩码按位或。
```

## `mask_xor`
**签名**：`mask_xor(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位异或。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t a_data[L] = {};
alignas(ksimd::alignment::Required) int32_t b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<int32_t>(i + 2); b_data[i] = static_cast<int32_t>(i + 1); }
auto va = ns::loadu(tag, a_data);
auto vb = ns::loadu(tag, b_data);
auto ma = ns::greater(tag, va, vb);
auto mb = ns::less_equal(tag, va, vb);
// 示例代码:
auto mask = ns::mask_xor(tag, ma, mb);
// 效果:
// 掩码按位异或。
```

## `mask_not`
**签名**：`mask_not(Tag tag, Mask<Tag> mask)`  
**参数**：`tag: Tag (any)`，`mask: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码按位取反。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
uint8_t a[L] = {}; uint8_t b[L] = {};
for (size_t i = 0; i < L; ++i) { a[i] = static_cast<uint8_t>(i + 1); b[i] = static_cast<uint8_t>(i); }
auto va = ns::loadu(tag, a);
auto vb = ns::loadu(tag, b);
auto m = ns::greater(tag, va, vb);
// 示例代码:
auto result = ns::mask_not(tag, m);
// 效果:
// 掩码按位取反。
```

## `mask_and_not`
**签名**：`mask_and_not(Tag tag, Mask<Tag> lhs, Mask<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Mask<Tag>`，`rhs: Mask<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：掩码 `~lhs & rhs`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t a_data[L] = {};
alignas(ksimd::alignment::Required) int32_t b_data[L] = {};
for (size_t i = 0; i < L; ++i) { a_data[i] = static_cast<int32_t>(i + 2); b_data[i] = static_cast<int32_t>(i + 1); }
auto va = ns::loadu(tag, a_data);
auto vb = ns::loadu(tag, b_data);
auto ma = ns::greater(tag, va, vb);
auto mb = ns::less_equal(tag, va, vb);
// 示例代码:
auto mask = ns::mask_and_not(tag, ma, mb);
// 效果:
// 掩码 `~lhs & rhs`。
```

## `mask_all`
**签名**：`mask_all(Tag tag, Mask<Tag> mask)`  
**参数**：`tag: Tag (any)`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：所有 lane 是否都为真。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
float a[L] = {}; float b[L] = {};
for (size_t i = 0; i < L; ++i) { a[i] = 1.0f; b[i] = 0.0f; }
auto va = ns::loadu(tag, a);
auto vb = ns::loadu(tag, b);
auto m = ns::greater(tag, va, vb);
// 示例代码:
bool result = ns::mask_all(tag, m);
// 效果:
// 所有 lane 是否都为真。
```

## `mask_any`
**签名**：`mask_any(Tag tag, Mask<Tag> mask)`  
**参数**：`tag: Tag (any)`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：是否存在任一 lane 为真。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
double a[L] = {}; double b[L] = {};
for (size_t i = 0; i < L; ++i) { a[i] = static_cast<double>(i + 2); b[i] = static_cast<double>(i + 1); }
auto va = ns::loadu(tag, a);
auto vb = ns::loadu(tag, b);
auto m = ns::greater(tag, va, vb);
// 示例代码:
bool result = ns::mask_any(tag, m);
// 效果:
// 是否存在任一 lane 为真。
```

## `mask_none`
**签名**：`mask_none(Tag tag, Mask<Tag> mask)`  
**参数**：`tag: Tag (any)`，`mask: Mask<Tag>`  
**返回类型**：`bool`  
**返回语义**：是否所有 lane 都为假。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = 0;
auto v = ns::loadu(tag, data);
auto mask = ns::greater(tag, v, v);
// 示例代码:
bool result = ns::mask_none(tag, mask);
// 效果:
// 是否所有 lane 都为假。
```

## `if_then_else`
**签名**：`if_then_else(Tag tag, Mask<Tag> _if, Batch<Tag> _then, Batch<Tag> _else)`  
**参数**：`tag: Tag (any)`，`_if: Mask<Tag>`，`_then: Batch<Tag>`，`_else: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：按 `_if` 掩码逐 lane 选择 `_then` 或 `_else`。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
uint8_t a[L] = {}; uint8_t b[L] = {};
for (size_t i = 0; i < L; ++i) { a[i] = static_cast<uint8_t>(i + 1); b[i] = 0; }
auto va = ns::loadu(tag, a);
auto vb = ns::loadu(tag, b);
auto cond = ns::greater(tag, va, vb);
// 示例代码:
auto result = ns::if_then_else(tag, cond, va, vb);
// 效果:
// 按 `_if` 掩码逐 lane 选择 `_then` 或 `_else`。
```

## `reduce_add`
**签名**：`reduce_add(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>` 或 `int32_t`（依类型分支）  
**返回语义**：将所有 lane 求和并返回标量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = static_cast<int32_t>(i + 1);
auto v = ns::loadu(tag, data);
// 示例代码:
auto result = ns::reduce_add(tag, v);
// 效果:
// 将所有 lane 求和并返回标量。
```

## `reduce_mul`
**签名**：`reduce_mul(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>` 或 `int32_t`（依类型分支）  
**返回语义**：将所有 lane 连乘并返回标量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<uint8_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
uint8_t data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = static_cast<uint8_t>((i % 3) + 1);
auto v = ns::loadu(tag, data);
// 示例代码:
auto result = ns::reduce_mul(tag, v);
// 效果:
// 将所有 lane 连乘并返回标量。
```

## `reduce_min`
**签名**：`reduce_min(Tag tag, Batch<Tag> v)`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>`  
**返回语义**：返回 `v` 所有 lane 的最小值标量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
double data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = static_cast<double>(L - i);
auto v = ns::loadu(tag, data);
// 示例代码:
auto result = ns::reduce_min<ns::FloatMinMaxOption::Native>(tag, v);
// 效果:
// 返回 `v` 所有 lane 的最小值标量。
```

## `reduce_max`
**签名**：`reduce_max(Tag tag, Batch<Tag> v)`  
**参数**：`option: FloatMinMaxOption`，`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`tag_scalar_t<Tag>`  
**返回语义**：返回 `v` 所有 lane 的最大值标量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
int32_t data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = static_cast<int32_t>(i - 3);
auto v = ns::loadu(tag, data);
// 示例代码:
auto result = ns::reduce_max<ns::FloatMinMaxOption::Native>(tag, v);
// 效果:
// 返回 `v` 所有 lane 的最大值标量。
```

## `abs`
**签名**：`abs(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 绝对值结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<int32_t>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::abs(tag, v);
// 效果:
// 逐 lane 绝对值结果。
```

## `neg`
**签名**：`neg(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 取负结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<int32_t>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) int32_t in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<int32_t>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::neg(tag, v);
// 效果:
// 逐 lane 取负结果。
```

## `div`
**签名**：`div(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 浮点除法 `lhs/rhs` 结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 1); rhs_data[i] = static_cast<double>(i + 3); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto result = ns::div(tag, lhs, rhs);
// 效果:
// 逐 lane 浮点除法 `lhs/rhs` 结果。
```

## `sqrt`
**签名**：`sqrt(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 平方根结果。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<float>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::sqrt(tag, v);
// 效果:
// 逐 lane 平方根结果。
```

## `round`
**签名**：`round(Tag tag, Batch<Tag> v)`  
**参数**：`mode: RoundingMode`，`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：按 `mode` 对每个 lane 舍入后的向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
double data[L] = {};
for (size_t i = 0; i < L; ++i) data[i] = static_cast<double>(i) + 0.5;
auto v = ns::loadu(tag, data);
// 示例代码:
auto result = ns::round<ns::RoundingMode::Nearest>(tag, v);
// 效果:
// 按 `mode` 对每个 lane 舍入后的向量。
```

## `not_greater`
**签名**：`not_greater(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非大于”判断掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_greater(tag, lhs, rhs);
// 效果:
// 逐 lane “非大于”判断掩码。
```

## `not_greater_equal`
**签名**：`not_greater_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非大于等于”判断掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 2); rhs_data[i] = static_cast<double>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_greater_equal(tag, lhs, rhs);
// 效果:
// 逐 lane “非大于等于”判断掩码。
```

## `not_less`
**签名**：`not_less(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非小于”判断掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_less(tag, lhs, rhs);
// 效果:
// 逐 lane “非小于”判断掩码。
```

## `not_less_equal`
**签名**：`not_less_equal(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane “非小于等于”判断掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 2); rhs_data[i] = static_cast<double>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_less_equal(tag, lhs, rhs);
// 效果:
// 逐 lane “非小于等于”判断掩码。
```

## `any_NaN`
**签名**：`any_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs`/`rhs` 任一为 NaN 的掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::any_NaN(tag, lhs, rhs);
// 效果:
// 逐 lane 判断 `lhs`/`rhs` 任一为 NaN 的掩码。
```

## `all_NaN`
**签名**：`all_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 与 `rhs` 均为 NaN 的掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 2); rhs_data[i] = static_cast<double>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::all_NaN(tag, lhs, rhs);
// 效果:
// 逐 lane 判断 `lhs` 与 `rhs` 均为 NaN 的掩码。
```

## `not_NaN`
**签名**：`not_NaN(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断两者都不是 NaN 的掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::not_NaN(tag, lhs, rhs);
// 效果:
// 逐 lane 判断两者都不是 NaN 的掩码。
```

## `any_finite`
**签名**：`any_finite(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 或 `rhs` 任一为有限值的掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<double>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) double lhs_data[L] = {};
alignas(ksimd::alignment::Required) double rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<double>(i + 2); rhs_data[i] = static_cast<double>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::any_finite(tag, lhs, rhs);
// 效果:
// 逐 lane 判断 `lhs` 或 `rhs` 任一为有限值的掩码。
```

## `all_finite`
**签名**：`all_finite(Tag tag, Batch<Tag> lhs, Batch<Tag> rhs)`  
**参数**：`tag: Tag (any)`，`lhs: Batch<Tag>`，`rhs: Batch<Tag>`  
**返回类型**：`Mask<Tag>`  
**返回语义**：逐 lane 判断 `lhs` 与 `rhs` 均为有限值的掩码。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float lhs_data[L] = {};
alignas(ksimd::alignment::Required) float rhs_data[L] = {};
for (size_t i = 0; i < L; ++i) { lhs_data[i] = static_cast<float>(i + 2); rhs_data[i] = static_cast<float>(i + 1); }
// 示例代码:
auto lhs = ns::loadu(tag, lhs_data);
auto rhs = ns::loadu(tag, rhs_data);
auto mask = ns::all_finite(tag, lhs, rhs);
// 效果:
// 逐 lane 判断 `lhs` 与 `rhs` 均为有限值的掩码。
```

## `rcp`
**签名**：`rcp(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 倒数近似值向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<float>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::rcp(tag, v);
// 效果:
// 逐 lane 倒数近似值向量。
```

## `rsqrt`
**签名**：`rsqrt(Tag tag, Batch<Tag> v)`  
**参数**：`tag: Tag (any)`，`v: Batch<Tag>`  
**返回类型**：`Batch<Tag>`  
**返回语义**：逐 lane 逆平方根近似值向量。
```c++
// 输入:
namespace ns = ksimd::KSIMD_DYN_INSTRUCTION;
auto tag = ns::FullTag<float>{};
constexpr size_t L = ns::lanes(tag); // for ARM SVE, it is NOT a constexpr variable
alignas(ksimd::alignment::Required) float in_data[L] = {};
for (size_t i = 0; i < L; ++i) in_data[i] = static_cast<float>(i + 1);
// 示例代码:
auto v = ns::loadu(tag, in_data);
auto result = ns::rsqrt(tag, v);
// 效果:
// 逐 lane 逆平方根近似值向量。
```
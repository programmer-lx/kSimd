#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <numeric>

#include "../test.hpp"
#include "kSimd/kernels/popcnt/popcnt.h"

template<typename T>
    requires (std::is_integral_v<T> && !std::is_signed_v<T>)
ks_bitcount_t test_popcnt_ref(T x)
{
    ks_bitcount_t cnt = 0;
    for (int i = 0; i < (int)(sizeof(T) * 8); ++i)
    {
        // 使用 T(1) 确保位移宽度与 T 一致
        // 判断非零即加 1
        if ((x >> i) & T(1))
        {
            cnt++;
        }
    }
    return cnt;
}

TEST(popcnt, intrinsic_support)
{
    const ksimd::CpuSupportInfo info = ksimd::get_cpu_support_info();
    EXPECT_TRUE(info.POPCNT);
}

TEST(popcnt, popcnt8_soft)
{
    // 穷举 0 - 255
    for (int i = 0; i <= 255; ++i)
    {
        uint8_t val = static_cast<uint8_t>(i);

        ks_bitcount_t expected = test_popcnt_ref(val);

        EXPECT_EQ(ks_popcnt8_soft(val), expected)
            << "Failed at value: " << i;
    }
}

TEST(popcnt, popcnt16_soft)
{
    // u16 只有 65536 种情况，可以直接全量穷举
    for (int i = 0; i <= 0xFFFF; ++i)
    {
        uint16_t val = static_cast<uint16_t>(i);
        ks_bitcount_t expected = test_popcnt_ref(val);

        EXPECT_EQ(ks_popcnt16_soft(val), expected)
            << "Failed at value: 0x" << std::hex << i;
    }
}

TEST(popcnt, popcnt32_soft)
{
    // 1. 基础边界测试
    EXPECT_EQ(ks_popcnt32_soft(0u), 0u);
    EXPECT_EQ(ks_popcnt32_soft(0xFFFFFFFFu), 32u);
    EXPECT_EQ(ks_popcnt32_soft(1u), 1u);
    EXPECT_EQ(ks_popcnt32_soft(1u << 31), 1u);

    // 2. 特殊位模式测试
    EXPECT_EQ(ks_popcnt32_soft(0x55555555u), 16u); // 0101...
    EXPECT_EQ(ks_popcnt32_soft(0x33333333u), 16u); // 0011...
    EXPECT_EQ(ks_popcnt32_soft(0x0F0F0F0Fu), 16u); // 00001111...

    // 3. 随机采样对比测试 (测试 100,000 个随机数)
    std::mt19937 gen(42); // 固定种子保证测试可复现
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFFu);

    for (int i = 0; i < 100000; ++i)
    {
        uint32_t val = dis(gen);
        ks_bitcount_t expected = test_popcnt_ref(val);

        ASSERT_EQ(ks_popcnt32_soft(val), expected)
            << "Failed at random value: 0x" << std::hex << val;
    }
}

TEST(popcnt, popcnt64_soft)
{
    // 1. 基础边界测试
    EXPECT_EQ(ks_popcnt64_soft(0), 0u);
    EXPECT_EQ(ks_popcnt64_soft(0xFFFFFFFFFFFFFFFFull), 64u);
    EXPECT_EQ(ks_popcnt64_soft(1ull), 1u);
    EXPECT_EQ(ks_popcnt64_soft(1ull << 63), 1u);

    // 2. 特殊位模式测试
    EXPECT_EQ(ks_popcnt64_soft(0x5555555555555555ull), 32u); // 0101...
    EXPECT_EQ(ks_popcnt64_soft(0x3333333333333333ull), 32u); // 0011...
    EXPECT_EQ(ks_popcnt64_soft(0x0F0F0F0F0F0F0F0Full), 32u); // 00001111...

    // 3. 随机/复杂模式对比测试
    uint64_t test_cases[] = {
        0x123456789ABCDEF0ull,
        0xDEADBEEFCAFEBABEull,
        0x8000000000000001ull,
        UINT64_C(0x1111111111111111)
    };

    for (uint64_t val : test_cases)
    {
        ks_bitcount_t expected = test_popcnt_ref(val);
        EXPECT_EQ(ks_popcnt64_soft(val), static_cast<size_t>(expected))
            << "Failed at value: 0x" << std::hex << val;
    }
}

static ks_bitcount_t reference_popcnt(const void* buffer, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(buffer);
    ks_bitcount_t count = 0;
    for (size_t i = 0; i < size; ++i) {
        count += test_popcnt_ref(p[i]);
    }
    return count;
}

TEST(popcnt, popcnt_buffer_soft)
{
    // 1. 基础边界测试
    EXPECT_EQ(ks_test_popcnt_buffer_soft(nullptr, 0), 0u);

    // 2. 各种长度的对齐与非对齐测试 (1 到 67 字节，覆盖 8 字节边界)
    for (size_t len = 1; len <= 67; ++len) {
        std::vector<uint8_t> data(len, 0xFF); // 全 1 填充
        size_t expected = len * 8;
        EXPECT_EQ(ks_test_popcnt_buffer_soft(data.data(), len), expected) 
            << "Failed at full-ones length: " << len;

        std::vector<uint8_t> zero_data(len, 0); // 全 0 填充
        EXPECT_EQ(ks_test_popcnt_buffer_soft(zero_data.data(), len), 0u) 
            << "Failed at zero length: " << len;
    }

    // 3. 随机数据大压力测试
    std::mt19937 gen(42); 
    std::uniform_int_distribution<uint32_t> dis(0, 255);
    
    const size_t large_size = 1024 + 7; // 故意选一个非对齐的大小
    std::vector<uint8_t> random_buffer(large_size);
    for(auto& b : random_buffer) b = (uint8_t)dis(gen);

    ks_bitcount_t expected_rand = reference_popcnt(random_buffer.data(), large_size);
    EXPECT_EQ(ks_test_popcnt_buffer_soft(random_buffer.data(), large_size), expected_rand)
        << "Failed at large random buffer test";
}

TEST(popcnt, x86)
{
    {
        // 1. 空缓冲区测试
        EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(nullptr, 0), 0u);

        // 2. 覆盖所有阶梯长度 (从 0 字节到 32 字节，包含 4 字节和 8 字节切换点)
        for (size_t len = 0; len <= 32; ++len) {
            std::vector<uint8_t> buffer(len);

            // 测试全 1 场景
            std::fill(buffer.begin(), buffer.end(), 0xFF);
            EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(buffer.data(), len), len * 8)
                << "Failed at full-ones, length: " << len;

            // 测试全 0 场景
            std::fill(buffer.begin(), buffer.end(), 0x00);
            EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(buffer.data(), len), 0u)
                << "Failed at zeros, length: " << len;

            // 测试交替位场景
            std::fill(buffer.begin(), buffer.end(), 0x55); // 01010101
            EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(buffer.data(), len), len * 4)
                << "Failed at 0x55, length: " << len;
        }

        // 3. 随机数据大压力测试
        std::mt19937 gen(12345);
        std::uniform_int_distribution<int> dist(0, 255);

        // 选一个较大的、非对齐的长度（如 1KB + 7字节）
        const size_t large_len = 1024 + 7;
        std::vector<uint8_t> large_buffer(large_len);
        for (auto& b : large_buffer) b = static_cast<uint8_t>(dist(gen));

        ks_bitcount_t expected = reference_popcnt(large_buffer.data(), large_len);
        EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(large_buffer.data(), large_len), expected)
            << "Failed at large random buffer";
    }

    {
        // 4. 对齐偏移测试（验证 std::memcpy 是否真的解决了非对齐读取问题）
        const char* pattern = "HighPerformanceComputing"; // 24 bytes
        size_t pattern_len = strlen(pattern);

        // 我们故意在不同的偏移量上调用函数
        ks_bitcount_t ref_val = reference_popcnt(pattern, pattern_len);

        // 模拟从非对齐地址开始读取
        EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(pattern, pattern_len), ref_val);
        EXPECT_EQ(ks_test_popcnt_buffer_x86_popcnt(pattern + 1, pattern_len - 1),
                  reference_popcnt(pattern + 1, pattern_len - 1));
    }
}

TEST(popcnt, speed_test)
{
    // 1. 准备大规模数据 (比如 128MB)
    const size_t buffer_size = 30 * 1024 * 1024;
    std::vector<uint8_t> buffer(buffer_size);

    // 填充伪随机数据
    std::mt19937 gen(42);
    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i] = static_cast<uint8_t>(gen() & 0xFF);
    }

    const int iterations = 100; // 循环执行 100 次以获得更稳定的结果
    size_t total_ones = 0;

    std::cout << "Starting benchmark: 30MB buffer, " << iterations << " iterations..." << std::endl;

    {
        // 2. 创建计时器，作用域开始
        ScopeTimer timer("x86_popcnt_buffer");

        for (int i = 0; i < iterations; ++i) {
            // 每次强制使用结果，防止编译器把循环优化掉
            total_ones += ks_test_popcnt_buffer_x86_popcnt(buffer.data(), buffer_size);
        }
    }
    {
        ScopeTimer timer("popcnt_buffer_soft");

        for (int i = 0; i < iterations; ++i) {
            // 每次强制使用结果，防止编译器把循环优化掉
            total_ones += ks_test_popcnt_buffer_soft(buffer.data(), buffer_size);
        }
    }

    // 打印总数只是为了确保运算没被优化
    std::cout << "Total ones counted: " << total_ones << std::endl;
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

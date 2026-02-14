#include "kSimd/kernels/crc32c/crc32c.h"

#include "kSimd/core/impl/base.hpp"

#include <gtest/gtest.h>


TEST(crc32c, checksum_test_empty)
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, size);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, size);;
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_one_byte_zero)
{
    uint8_t data[] = {0x00};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 1);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 1);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_one_byte_ff)
{
    uint8_t data[] = {0xFF};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 1);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 1);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_pow2_4bytes)
{
    uint8_t data[] = {0,1,2,3};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 4);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 4);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_pow2_16bytes)
{
    uint8_t data[16];
    for (int i = 0; i < 16; ++i) data[i] = uint8_t(i);

    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 16);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 16);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_pow2_32bytes)
{
    uint8_t data[32];
    for (int i = 0; i < 32; ++i) data[i] = uint8_t(i);

    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 32);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 32);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_non_pow2_3bytes)
{
    uint8_t data[] = {0x11, 0x22, 0x33};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 3);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 3);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_non_pow2_7bytes)
{
    uint8_t data[] = {1,2,3,4,5,6,7};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 7);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 7);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_all_zero_64)
{
    uint8_t data[64] = {};
    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 64);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 64);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_all_ff_64)
{
    uint8_t data[64];
    memset(data, 0xFF, 64);

    auto soft = ks_begin_crc32c();
    soft = ks_test_update_crc32c_soft(soft, data, 64);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_begin_crc32c();
    x86 = ks_test_update_crc32c_sse42(x86, data, 64);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_non_zero_origin)
{
    uint8_t data[] = {1,2,3,4,5};

    auto soft = ks_test_update_crc32c_soft(0xFFFFFFFF, data, 5);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_test_update_crc32c_sse42(0xFFFFFFFF, data, 5);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_chunk_equivalence)
{
    uint8_t data[32];
    for (int i = 0; i < 32; ++i) data[i] = uint8_t(i);

    auto full_soft = ks_begin_crc32c();
    full_soft = ks_test_update_crc32c_soft(full_soft, data, 32);
    full_soft = ks_end_crc32c(full_soft);

    auto part1 = ks_test_update_crc32c_soft(ks_begin_crc32c(), data, 16);
    auto part2 = ks_test_update_crc32c_soft(part1, data + 16, 16);
    part2 = ks_end_crc32c(part2);

    EXPECT_TRUE(full_soft == part2);

#if KSIMD_ARCH_X86_ANY
    auto full_x86 = ks_test_update_crc32c_sse42(ks_begin_crc32c(), data, 32);
    full_x86 = ks_end_crc32c(full_x86);
    EXPECT_TRUE(full_soft == full_x86);
#endif
}

TEST(crc32c, checksum_test_unaligned_pointer)
{
    uint8_t buffer[65];
    for (int i = 0; i < 65; ++i) buffer[i] = uint8_t(i);

    auto soft = ks_test_update_crc32c_soft(ks_begin_crc32c(), buffer + 1, 64);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_test_update_crc32c_sse42(ks_begin_crc32c(), buffer + 1, 64);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}

TEST(crc32c, checksum_test_large_1024)
{
    uint8_t data[1024];
    for (int i = 0; i < 1024; ++i) data[i] = uint8_t(i * 7);

    auto soft = ks_test_update_crc32c_soft(ks_begin_crc32c(), data, 1024);
    soft = ks_end_crc32c(soft);

#if KSIMD_ARCH_X86_ANY
    auto x86 = ks_test_update_crc32c_sse42(ks_begin_crc32c(), data, 1024);
    x86 = ks_end_crc32c(x86);
    EXPECT_TRUE(soft == x86);
#endif
}


inline uint32_t ks_crc32c_oneshot_soft(const void* data, size_t size) {
    uint32_t crc = ks_begin_crc32c();
    crc = ks_test_update_crc32c_soft(crc, (const uint8_t*)data, size);
    return ks_end_crc32c(crc);
}

inline uint32_t ks_crc32c_oneshot_sse42(const void* data, size_t size) {
    uint32_t crc = ks_begin_crc32c();
    crc = ks_test_update_crc32c_sse42(crc, (const uint8_t*)data, size);
    return ks_end_crc32c(crc);
}

TEST(crc32c, standard_values) {
    // 1. 空字符串
    EXPECT_EQ(ks_crc32c_oneshot_soft("", 0), 0x00000000);

    // 2. 数字 1-9 (这是最经典的验证点)
    EXPECT_EQ(ks_crc32c_oneshot_soft("123456789", 9), 0xE3069283);

    // 4. 32 字节全 0x00
    std::vector<uint8_t> zeros(32, 0);
    EXPECT_EQ(ks_crc32c_oneshot_soft(zeros.data(), 32), 0x8A9136AA);
}

int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#include <vector>
#include <iostream>

#include "kSimd/kernels/crc32c/crc32c.h"

#include "utils.hpp"

int main()
{
    constexpr size_t NUM = 100 * 1024 * 1024 + 3; // 100MB + 3B (unaligned)
    std::vector<uint8_t> buffer(NUM);
    for (size_t i = 0; i < NUM; ++i)
    {
        buffer[i] = static_cast<uint8_t>(random_f(-254.0f, 254.0f));
    }

    // 预热
    uint32_t checksum = ks_update_crc32c(ks_begin_crc32c(), buffer.data() + 3, NUM - 3);
    checksum = ks_end_crc32c(checksum);
    [[maybe_unused]] volatile void* unused_ptr = &checksum;

    // 计时
    {
        ScopeTimer timer("crc32c");
        checksum = ks_begin_crc32c();
        checksum = ks_update_crc32c(checksum, buffer.data() + 3, NUM - 3);
        checksum = ks_end_crc32c(checksum);

        std::cout << std::format("checksum: {}", checksum) << std::endl;
    }

    // 验证已知的CRC32C
    {
        checksum = ks_begin_crc32c();
        const char* str = "123456789";
        checksum = ks_update_crc32c(checksum, reinterpret_cast<const uint8_t*>(str), 9);
        checksum = ks_end_crc32c(checksum);

        if (checksum == UINT32_C(0xE3069283))
        {
            std::cout << "123456789 CRC32C = 0xE3069283, correct!\n";
        }
        else
        {
            std::cout << "Incorrect CRC32C checksum!\n";
        }
    }

    return 0;
}
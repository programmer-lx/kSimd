#include <vector>
#include <iostream>

#include "kSimd/kernels/crc32c/crc32c.h"

#include "utils.hpp"

int main()
{
    constexpr size_t NUM = 30 * 1024 * 1024 + 3; // 30MB + 3B (unaligned)
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
        checksum = ks_update_crc32c(ks_begin_crc32c(), buffer.data() + 3, NUM - 3);
        checksum = ks_end_crc32c(checksum);

        std::cout << std::format("checksum: {}", checksum) << std::endl;
    }

    return 0;
}
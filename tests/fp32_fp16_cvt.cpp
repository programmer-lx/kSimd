#include <vector>
#include <cmath>
#include <limits>

#include <kSimd/base_op.hpp>
// #include <kSimd/fp16_convert.hpp>

#include "test.hpp"


int main(int argc, char **argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
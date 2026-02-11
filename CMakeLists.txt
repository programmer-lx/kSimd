cmake_minimum_required(VERSION 3.22.1)
project(
        kSimd
        LANGUAGES CXX
        VERSION 0.0.0.1
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(KSIMD_BUILD_EXAMPLES "" OFF)
option(KSIMD_BUILD_TESTS "" OFF)


# kSimd library (header-only)
add_library(kSimd INTERFACE)
target_include_directories(
        kSimd INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FP16/include
)

if(KSIMD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(KSIMD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

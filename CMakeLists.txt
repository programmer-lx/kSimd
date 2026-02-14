cmake_minimum_required(VERSION 3.22.1)
project(
        kSimd
        LANGUAGES CXX
        VERSION 0.0.0.1
)

# options
option(KSIMD_BUILD_EXAMPLES "" OFF)
option(KSIMD_BUILD_TESTS "" OFF)


# kSimd library (header-only)
add_library(kSimd INTERFACE)
target_include_directories(
        kSimd INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/kSimd
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FP16/include
)
target_compile_features(kSimd INTERFACE cxx_std_20)

if(KSIMD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(KSIMD_BUILD_TESTS)
    set(KSIMD_SOURCE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    enable_testing()
    add_subdirectory(tests)
endif()

cmake_minimum_required(VERSION 3.22.1)
project(
        kSimd
        LANGUAGES CXX
        VERSION 0.0.0.1
)

# options
option(KSIMD_BUILD_EXAMPLES "" OFF)
option(KSIMD_BUILD_TESTS "" OFF)


# static library (with one cpp to detect cpu features)
add_library(kSimd STATIC ${CMAKE_CURRENT_SOURCE_DIR}/kSimd/kSimd/core/one_cpp.cpp)
if(MSVC)
    target_compile_options(kSimd PRIVATE /utf-8 /W4 /WX /EHs-c- /GR-)
    target_link_options(kSimd PRIVATE /WX)
else()
    target_compile_options(kSimd PRIVATE -fno-exceptions -fno-rtti -Werror -Wall)
endif()
target_include_directories(
        kSimd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/kSimd
)
target_compile_features(kSimd PUBLIC cxx_std_20)

if(KSIMD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(KSIMD_BUILD_TESTS)
    set(KSIMD_SOURCE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    enable_testing()
    add_subdirectory(tests)
endif()

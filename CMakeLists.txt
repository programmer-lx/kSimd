cmake_minimum_required(VERSION 3.22.1)
project(
        kSimd
        LANGUAGES CXX
        VERSION 0.0.0.1
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# options
option(KSIMD_BUILD_EXAMPLES "" OFF)
option(KSIMD_BUILD_TESTS "" OFF)


# kSimd library (static)
add_library(kSimd STATIC)
target_include_directories(kSimd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/FP16/include)
target_sources(kSimd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/one_cpp.cpp)
if(MSVC)
    target_compile_options(kSimd PRIVATE /utf-8)
    target_compile_definitions(kSimd PRIVATE UNICODE _UNICODE)

    target_compile_options(kSimd PRIVATE /W4 /WX /EHs-c-) # 禁用异常
    target_link_options(kSimd PRIVATE /WX)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(kSimd PRIVATE -Werror -Wall -fno-exceptions) # 禁用异常
endif()

if(KSIMD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


if(KSIMD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

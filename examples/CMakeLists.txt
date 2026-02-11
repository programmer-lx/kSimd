function(example_target_actions TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /Gv>)
        target_compile_options(${TARGET_NAME} PRIVATE /WX)
        target_link_options(${TARGET_NAME} PRIVATE /WX)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE $<$<CONFIG:Release>:-O2>)
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Werror)
#        target_link_options(${TARGET_NAME} PRIVATE -Wl --fatal-warnings)
    endif()
endfunction()

set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_LINK_WARNING_AS_ERROR ON)

if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(UNICODE _UNICODE)
endif()

# dynamic dispatch
add_executable(example_simd_batch_dyn_dispatch simd_batch_dyn_dispatch.cpp)
target_link_libraries(example_simd_batch_dyn_dispatch PRIVATE kSimd)
if(MSVC)
    # 用于测试编译期分发表裁剪，在MSVC下不打开AVX2开关，会分发AVX2_MAX以下的指令集
#    target_compile_options(example_simd_batch_dyn_dispatch PRIVATE /arch:AVX2)
else()
    # GCC clang 打开 AVX2_MAX 开关，不分发比AVX2_MAX更低级的指令集了
    target_compile_options(example_simd_batch_dyn_dispatch PRIVATE -mavx2 -mfma -mf16c)
endif()
target_include_directories(example_simd_batch_dyn_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
example_target_actions(example_simd_batch_dyn_dispatch)
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
target_include_directories(example_simd_batch_dyn_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
example_target_actions(example_simd_batch_dyn_dispatch)

# static dispatch
add_executable(example_simd_batch_static_dispatch simd_batch_static_dispatch.cpp)
target_link_libraries(example_simd_batch_static_dispatch PRIVATE kSimd)
target_include_directories(example_simd_batch_static_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
example_target_actions(example_simd_batch_static_dispatch)
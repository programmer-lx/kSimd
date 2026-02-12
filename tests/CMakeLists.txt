function(add_extra_test_compile_and_link_options TARGET_NAME)
    if(MSVC)
        if(KSIMD_TEST_OPTION STREQUAL "od")
            target_compile_options(${TARGET_NAME} PRIVATE /Od) # Debug: Od
        endif()
        if(KSIMD_TEST_OPTION STREQUAL "o2")
            target_compile_options(${TARGET_NAME} PRIVATE /O2) # Release: O2
        endif()
        if(KSIMD_TEST_OPTION STREQUAL "gl")
            target_compile_options(${TARGET_NAME} PRIVATE /O2 /GL) # Release: O2 GL
            target_link_options(${TARGET_NAME} PRIVATE /LTCG)
        endif()
    else()
        if(KSIMD_TEST_OPTION STREQUAL "od")
            target_compile_options(${TARGET_NAME} PRIVATE -O0) # Debug: O0
        endif()
        if(KSIMD_TEST_OPTION STREQUAL "o2")
            target_compile_options(${TARGET_NAME} PRIVATE -O2) # Release: O2
        endif()
        if(KSIMD_TEST_OPTION STREQUAL "gl")
            target_compile_options(${TARGET_NAME} PRIVATE -O2 -flto) # Release: O2 -flto
            target_link_options(${TARGET_NAME} PRIVATE -flto=auto)
        endif()
    endif()
endfunction()

function(add_ksimd_test_target TARGET_NAME SOURCE_FILE)
    set(FINAL_TARGET_NAME ${TARGET_NAME}_${KSIMD_TEST_OPTION})
    add_executable(${FINAL_TARGET_NAME} ${SOURCE_FILE})
    if(MSVC)
        target_compile_options(${FINAL_TARGET_NAME} PRIVATE /utf-8 /W4 /WX)
        target_link_options(${FINAL_TARGET_NAME} PRIVATE /WX)
        target_compile_definitions(${FINAL_TARGET_NAME} PRIVATE UNICODE _UNICODE)
    else()
        target_compile_options(${FINAL_TARGET_NAME} PRIVATE -Werror -Wall)
    endif()
    add_extra_test_compile_and_link_options(${FINAL_TARGET_NAME})

#    target_compile_definitions(kSimd PUBLIC KSIMD_IS_TESTING)
    target_compile_definitions(${FINAL_TARGET_NAME} PRIVATE KSIMD_IS_TESTING)
    target_link_libraries(${FINAL_TARGET_NAME} PRIVATE gtest kSimd)
    target_include_directories(${FINAL_TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_features(${FINAL_TARGET_NAME} PRIVATE cxx_std_23) # 测试的时候，使用C++23的部分特性，但是库的最低标准是C++20

    get_target_property(OPTIONS ${FINAL_TARGET_NAME} COMPILE_OPTIONS)
    message(STATUS "${FINAL_TARGET_NAME} compile options: ${OPTIONS}")

    add_test(NAME ${FINAL_TARGET_NAME} COMMAND $<TARGET_FILE:${FINAL_TARGET_NAME}>)
endfunction()


include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/programmer-lx/googletest.git
        GIT_TAG        v1.17.0
)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)


file(GLOB_RECURSE TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

foreach(SOURCE_FILE ${TEST_SOURCES})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${SOURCE_FILE}")
    get_filename_component(TEST_TARGET_NAME ${TEST_TARGET_NAME} NAME_WE)

    add_ksimd_test_target(${TEST_TARGET_NAME} ${SOURCE_FILE})
endforeach()

# extra test

# inline test
add_executable(inline_test inline_test/inline_cpp_1.cc inline_test/inline_cpp_2.cc inline_test/inline_main.cc)
target_include_directories(inline_test PRIVATE inline_test)
target_compile_features(inline_test PRIVATE cxx_std_23)
add_test(NAME inline_test COMMAND COMMAND $<TARGET_FILE:inline_test>)

# static test
add_executable(static_test static_test/static_cpp_1.cc static_test/static_cpp_2.cc static_test/static_main.cc)
target_include_directories(static_test PRIVATE static_test)
target_compile_features(static_test PRIVATE cxx_std_23)
add_test(NAME static_test COMMAND COMMAND $<TARGET_FILE:static_test>)

# macro test
add_executable(macro_test macro_test/kernel_disable_avx2_max.cc macro_test/kernel_enable_all.cc macro_test/main.cc)
target_include_directories(macro_test PRIVATE macro_test)
target_link_libraries(macro_test PRIVATE kSimd)
target_compile_features(macro_test PRIVATE cxx_std_23)
add_test(NAME macro_test COMMAND COMMAND $<TARGET_FILE:macro_test>)

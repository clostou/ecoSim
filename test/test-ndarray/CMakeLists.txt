cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(testNdarray LANGUAGES CXX)

# 将所有源文件编译为单一可执行目标
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/test_m1.cpp
    src/test_m2.cpp
    src/test_m3.cpp
    src/test_m4.cpp
    src/test_m5.cpp
    src/test_m6.cpp
    src/test_final.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8 /EHsc)
endif()

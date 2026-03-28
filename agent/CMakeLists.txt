# CMakeLists.txt: agent 静态库
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(agent LANGUAGES CXX)

# 头文件（header-only 库，此处收集以便 IDE 识别）
file(GLOB AGENT_HEADERS include/*.hpp)

# 创建 INTERFACE 库（当前实体系统为 header-only）
add_library(agent INTERFACE)
target_include_directories(agent INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../misc
)
target_compile_features(agent INTERFACE cxx_std_17)

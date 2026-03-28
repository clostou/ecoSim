# CMakeLists.txt: agent 单元测试
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(test-agent LANGUAGES CXX)

add_executable(test_agent
    src/test_entity.cpp
)

target_link_libraries(test_agent PRIVATE agent)
target_compile_features(test_agent PRIVATE cxx_std_17)

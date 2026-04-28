# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.18 FATAL_ERROR)

project (ecoSim LANGUAGES CXX)

# 包含子项目。
add_subdirectory (main)
#add_subdirectory (agent)
add_subdirectory (test/test-sfml)
add_subdirectory (test/test-cuda)
add_subdirectory (test/test-ndarray)
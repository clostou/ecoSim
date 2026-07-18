# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.18 FATAL_ERROR)
project (ecoSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 包含子项目。
add_subdirectory (main)
#add_subdirectory (agent)
add_subdirectory (test/test-sfml)
add_subdirectory (test/test-cuda)
add_subdirectory (test/test-ndarray)
add_subdirectory (test/test-cutlass)
add_subdirectory (test/test-attnlstm)
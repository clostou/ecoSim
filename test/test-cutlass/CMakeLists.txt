cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(testCutlass LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---- CUTLASS (reuse external cache) ----
include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/external)
set(FETCHCONTENT_FULLY_DISCONNECTED ON)
FetchContent_Declare(
    CUTLASS
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.5.1
    GIT_SHALLOW ON
)
set(CUTLASS_ENABLE_EXAMPLES OFF)
set(CUTLASS_ENABLE_TESTS OFF)
set(CUTLASS_ENABLE_TOOLS OFF)
FetchContent_MakeAvailable(CUTLASS)

# ---- Executable ----
add_executable(${PROJECT_NAME} src/fnn_test.cu)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "75;80;86")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# ---- Include project-wide headers (ndarray, etc.) ----
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ---- NVCC flags ----
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    # nvcc -O3 maps to host /O2, which conflicts with Debug's /Od /RTC1
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# ---- Link CUDA + CUTLASS ----
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cublas
    CUDA::cufft
    CUDA::cudart
    CUTLASS
)

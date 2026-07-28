cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(testAttnLstm LANGUAGES CXX CUDA)

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

# ---- Common setup function ----
function(setup_test_target TARGET_NAME SRC_FILE)
    add_executable(${TARGET_NAME} ${SRC_FILE})
    set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES "75;80;86")
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/agent/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
        # nvcc -O3 maps to host /O2, which conflicts with Debug's /Od /RTC1
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        CUDA::cublas
        CUDA::cufft
        CUDA::cudart
        CUTLASS
    )
endfunction()

# ---- Single-step test ----
setup_test_target(${PROJECT_NAME} src/attnlstm_test.cu)

# ---- Multi-step BPTT test ----
setup_test_target(testAttnLstmMulti src/attnlstm_multi_test.cu)

# ---- Performance benchmark ----
setup_test_target(testAttnLstmBench src/attnlstm_bench.cu)

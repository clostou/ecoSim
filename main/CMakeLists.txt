# CMakeList.txt: ecoSim 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/external)    # 指定本地共享缓存目录
set(FETCHCONTENT_FULLY_DISCONNECTED ON)    # 强制使用本地缓存
#set(FETCHCONTENT_SOURCE_DIR_SFML ${CMAKE_SOURCE_DIR}/misc/SFML-master)    # 手动指定源码位置
#set(ENV{HTTP_PROXY} "127.0.0.1:7897")    # 设置下载代理
#set(ENV{HTTPS_PROXY} "127.0.0.1:7897")
set(FETCHCONTENT_QUIET FALSE)    # 关闭安静模式，打印详细信息
set(FETCHCONTENT_LOG_DOWNLOADS TRUE)    # 记录下载过程（CMake 3.14+）
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.2
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
set(CMAKE_EXE_LINKER_FLAGS -static)    # or "-static-libgcc -static-libstdc++"

# 将源代码添加到此项目的可执行文件。
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/event.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
#target_include_directories(${PROJECT_NAME} PRIVATE ../include)
target_link_libraries(${PROJECT_NAME} PRIVATE SFML::Graphics)

add_custom_command(
        TARGET ${PROJECT_NAME}
        COMMENT "Copy res directory"
        PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/res $<TARGET_FILE_DIR:${PROJECT_NAME}>/res
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ecoSim)    # 设置vs活动项目
cmake_minimum_required(VERSION 3.12)
project(yolo_refactor LANGUAGES CXX CUDA)

# Set C++17 as the standard for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build configuration
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# OpenMP for parallel execution (used with std::execution::par)
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Filesystem library support for C++17 (needed for older compilers)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FILESYSTEM_LIBRARY stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FILESYSTEM_LIBRARY c++fs)
else()
    set(FILESYSTEM_LIBRARY "")
endif()

# CUDA configuration
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# TensorRT configuration
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# OpenCV configuration
include_directories(${OpenCV_INCLUDE_DIRS})

# Protocol buffers
set(protobuf_files ${CMAKE_CURRENT_SOURCE_DIR}/msgs/objects.proto)

# Project includes
include_directories(include)

# Recursive include function for subdirectories
function(include_sub_directories_recursively root_dir)
    if(IS_DIRECTORY ${root_dir})
        message("include dir: " ${root_dir})
        include_directories(${root_dir})
    endif()
    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if(IS_DIRECTORY ${root_dir}/${sub})
            include_sub_directories_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

include_sub_directories_recursively(${PROJECT_SOURCE_DIR}/include/yaml_cpp/)

# Compiler-specific flags for C++17 and performance
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# CUDA settings for modern C++
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Source files
file(GLOB_RECURSE YOLO_SRCS ${PROJECT_SOURCE_DIR}/src/yolo/*.cu)
file(GLOB_RECURSE CAM_SRCS ${PROJECT_SOURCE_DIR}/src/camstream/*.cpp ${PROJECT_SOURCE_DIR}/src/yolo/*.cu)

# Build CUDA libraries
cuda_add_library(yolov5_model SHARED ${YOLO_SRCS})
cuda_add_library(camstream SHARED ${CAM_SRCS})

# Create interface library for common dependencies
add_library(my_libs INTERFACE)
target_link_libraries(my_libs INTERFACE 
    nvinfer 
    cudart 
    ${OpenCV_LIBS} 
    ${PROJECT_SOURCE_DIR}/src/yaml_cpp/libyaml-cpp.so.0.6
    ${FILESYSTEM_LIBRARY}
)

# Link libraries to CUDA targets
target_link_libraries(yolov5_model my_libs)
target_link_libraries(camstream my_libs)

# Build radar library
add_library(radar SHARED src/radar/radar.cpp)
target_compile_features(radar PRIVATE cxx_std_17)

# Build common utilities library
aux_source_directory(src/common SRC_LIST)
add_library(common SHARED ${SRC_LIST})
target_compile_features(common PRIVATE cxx_std_17)

# Main executable
cuda_add_executable(yolo_refactor src/main.cpp)

# Set C++17 features for the main executable
target_compile_features(yolo_refactor PRIVATE cxx_std_17)

# Protocol buffer generation
PROTOBUF_TARGET_CPP(yolo_refactor ${CMAKE_CURRENT_SOURCE_DIR}/msgs/ ${protobuf_files})

# Link all libraries to main executable
target_link_libraries(yolo_refactor 
    PRIVATE
    jsoncpp 
    protobuf::libprotobuf 
    radar 
    yolov5_model 
    camstream 
    common 
    Threads::Threads
    ${FILESYSTEM_LIBRARY}
)

# Additional linking for parallel execution support
if(OPENMP_FOUND)
    target_link_libraries(yolo_refactor PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optional: Enable Link Time Optimization for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set_property(TARGET yolo_refactor PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "IPO/LTO enabled for release build")
    else()
        message(STATUS "IPO/LTO not supported: ${ipo_error}")
    endif()
endif()

# Set runtime path for shared libraries
set_target_properties(yolo_refactor PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)

# Installation rules
install(TARGETS yolo_refactor
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install shared libraries
install(TARGETS yolov5_model camstream radar common
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Create a config file for easy debugging
file(WRITE ${CMAKE_BINARY_DIR}/compile_commands.json.in
    "Compile commands database for IDE integration"
)

# Print configuration summary
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "OpenMP support: ${OPENMP_FOUND}")
message(STATUS "Filesystem library: ${FILESYSTEM_LIBRARY}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "===============================")
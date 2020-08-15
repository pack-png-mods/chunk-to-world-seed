cmake_minimum_required(VERSION 3.10)
project(chunk_to_world_seed LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


include(CheckLanguage)
check_language(CUDA REQUIRED)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support")
else()
    message(STATUS "No CUDA support")
endif()

add_executable(chunk_to_world_seed main.cu)
target_compile_features(chunk_to_world_seed PUBLIC cxx_std_14)

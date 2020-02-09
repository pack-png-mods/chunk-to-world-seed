cmake_minimum_required(VERSION 3.12)
project(chunk_to_world_seed C)

set(CMAKE_C_STANDARD 11)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support")
else()
    message(STATUS "No CUDA support")
endif()

add_executable(chunk_to_world_seed main.cu)

cmake_minimum_required(VERSION 3.2)
project(gpu)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
# http://stackoverflow.com/questions/29121211/cuda-compilation-issue-with-cmake
# Archtecture may change depending on CUDA version, see
# http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

message("CUDA: ${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARIES}")
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_library(gpu SHARED
  chamfer_distance.cu
  fast_chamfer_distance.cu
  smooth_l1_chamfer_distance.cu
  max_distance.cu)
target_link_libraries(gpu ${CUDA_LIBRARIES})
add_subdirectory(tests)
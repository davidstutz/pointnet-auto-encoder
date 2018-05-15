cmake_minimum_required(VERSION 3.2)
project(cpu)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

add_library(cpu SHARED
  chamfer_distance.cpp
  smooth_l1_chamfer_distance.cpp
  max_distance.cpp)
add_subdirectory(tests)
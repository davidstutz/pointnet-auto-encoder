cmake_minimum_required(VERSION 3.2)
project(cpu)

include_directories(../)
add_executable(test_chamfer_distance test_chamfer_distance.cpp)
target_link_libraries(test_chamfer_distance gpu)

add_executable(test_fast_chamfer_distance test_fast_chamfer_distance.cpp)
target_link_libraries(test_fast_chamfer_distance gpu)

add_executable(test_max_distance test_max_distance.cpp)
target_link_libraries(test_max_distance gpu)
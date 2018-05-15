cmake_minimum_required(VERSION 3.2)
project(cpu)

include_directories(../)
add_executable(test_chamfer_distance test_chamfer_distance.cpp)
target_link_libraries(test_chamfer_distance cpu)

add_executable(test_max_distance test_max_distance.cpp)
target_link_libraries(test_max_distance cpu)
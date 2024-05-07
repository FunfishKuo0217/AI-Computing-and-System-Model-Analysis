cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(hw2-4-1 hw2-4-1.cpp)
target_link_libraries(hw2-4-1 "${TORCH_LIBRARIES}")
set_property(TARGET hw2-4-1 PROPERTY CXX_STANDARD 17)

add_executable(hw2-4-2 hw2-4-2.cpp)
target_link_libraries(hw2-4-2 "${TORCH_LIBRARIES}")
set_property(TARGET hw2-4-2 PROPERTY CXX_STANDARD 17)

add_executable(hw2-4-3 hw2-4-3.cpp)
target_link_libraries(hw2-4-3 "${TORCH_LIBRARIES}")
set_property(TARGET hw2-4-3 PROPERTY CXX_STANDARD 17)



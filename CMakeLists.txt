cmake_minimum_required(VERSION 3.17)
project(MatMul)

set(CMAKE_CXX_STANDARD 20)

add_executable(MatMul src/main.cpp src/mat_mul.h src/timing.h)
add_subdirectory(deps/Raftlib)
target_link_libraries(MatMul PUBLIC raft)
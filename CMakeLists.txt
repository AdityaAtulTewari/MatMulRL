cmake_minimum_required(VERSION 3.17)
project(MatMul)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)

pkg_check_modules(RAFT_LIB REQUIRED IMPORTED_TARGET raftlib)

add_executable(MatMul src/main.cpp src/mat_mul.h src/timing.h)


target_link_libraries(MatMul PUBLIC PkgConfig::RAFT_LIB)

cmake_minimum_required(VERSION 3.10)
project(MatMul)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)

pkg_check_modules(RAFT_LIB REQUIRED IMPORTED_TARGET raftlib)

add_executable(MatMul src/main.cpp src/mat_mul.h src/timing.h)

target_link_libraries(MatMul PUBLIC PkgConfig::RAFT_LIB)

pkg_check_modules(LIBVL IMPORTED_TARGET libvl)
if(PkgConfig::LIBVL_FOUND)
  set(VL 1)
  target_link_libraries(MatMUL PUBLIC PkgConfig::LIBVL)
endif()


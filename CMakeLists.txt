cmake_minimum_required(VERSION 3.5)

project(Ising LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Ising main.cpp)

add_executable(2DIsing ising2d.cpp)
add_executable(3DIsing ising3d.cpp)

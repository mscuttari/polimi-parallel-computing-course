cmake_minimum_required(VERSION 3.10)
project(Particles_Serial C)

set(CMAKE_C_STANDARD 11)

add_executable(Particles_Serial Particles2D_2024.c)
target_link_libraries(Particles_Serial m)

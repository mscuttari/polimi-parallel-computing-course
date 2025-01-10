cmake_minimum_required(VERSION 3.10)
project(Particles_OpenMP C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(Particles_OpenMP Particles2D_2024.c)
target_link_libraries(Particles_OpenMP m MPI::MPI_C OpenMP::OpenMP_C)

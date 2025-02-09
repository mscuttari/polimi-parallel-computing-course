cmake_minimum_required(VERSION 3.10)
project(Particles_CUDA CUDA C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(Particles_CUDA Particles2D_2024.cu)
target_link_libraries(Particles_CUDA m MPI::MPI_C OpenMP::OpenMP_C)

if (OpenMP_FOUND)
    target_compile_options(Particles_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
endif()

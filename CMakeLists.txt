cmake_minimum_required(VERSION 3.18)
project(OpticalFlowCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Remember to add .cu files under src/ 
add_executable(optical_flow
    src/main.cu
    src/flow/horn_schunck.cu
)

target_include_directories(optical_flow PRIVATE include)
set_target_properties(optical_flow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

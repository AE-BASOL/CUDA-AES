cmake_minimum_required(VERSION 3.28)
project(CudaProject LANGUAGES CXX CUDA)

# Require CUDA and set standards
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Target your GPU arch (8.6 for RTX 3050 Ti)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Gather all .cu files in project directory
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/*.cu")

# Define a single executable
add_executable(CudaProject ${CUDA_SOURCES})

# Enable separable compilation so each .cu can compile alone
set_target_properties(CudaProject PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Include current dir for headers
target_include_directories(CudaProject PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against the CUDA runtime
target_link_libraries(CudaProject PRIVATE
        CUDA::cudart
)

target_include_directories(CudaProject PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include"
)

target_compile_definitions(CudaProject PRIVATE ENABLE_NVTX)

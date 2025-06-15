cmake_minimum_required(VERSION 3.28)
project(CudaProject LANGUAGES CXX CUDA)

# Require CUDA and set standards
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17) # Changed from 14 to 17

# Target your GPU arch (8.6 for RTX 3050 Ti)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Explicitly list CUDA source files
set(CUDA_SOURCES
    aes_tables.cu
    aes128_ctr.cu
    aes128_gcm.cu
    aes256_ctr.cu
    aes256_gcm.cu
    aes128_ecb.cu
    aes256_ecb.cu
    main.cu
)

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


# 1. CUDA derleyicisine özel optimize bayrakları
target_compile_options(CudaProject PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo --ptxas-options=-v>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# 2. Host (C++) tarafı için genel optimizasyon
target_compile_options(CudaProject PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 /EHsc>
)

# 3. (Opsiyonel) Kayıt sayısını sınırlayarak ilave latency azaltma
set_property(TARGET CudaProject PROPERTY CUDA_ARCHITECTURES 86)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES
        CUDA_MAXIMUM_REGISTER_COUNT 64)  # ya da 128


# In your CMakeLists.txt
include_directories("C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/include")

target_link_libraries(CudaProject PRIVATE
        "C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/libssl.lib"
        "C:/Users/efebasol/CLionProjects/aes_openSSL/openssl-3.3.3/libcrypto.lib"
        crypt32.lib       # for Cert* functions
        ws2_32.lib        # for socket-related functions
        user32.lib
        gdi32.lib
        advapi32.lib
        kernel32.lib)
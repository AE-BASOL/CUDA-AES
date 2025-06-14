cmake_minimum_required(VERSION 3.25)
project(CUDA_AES LANGUAGES CUDA C CXX)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Split per flag with list() to prevent bad quoting
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_DEBUG_FLAGS
            "-Xcompiler=/Od"
            "-Xcompiler=/Zi"
            "-Xcompiler=/RTC1"
            "-Xcompiler=/MDd"
            "-G"
    )
    set(CMAKE_CUDA_FLAGS_DEBUG "${CUDA_DEBUG_FLAGS}")
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
endif()

add_executable(cuda_aes AES_final.cu)
target_include_directories(cuda_aes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

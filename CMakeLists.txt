cmake_minimum_required(VERSION 3.28)

if(WIN32 AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "If nvcc cannot find cl.exe, run CMake from a Visual Studio Developer Command Prompt or pass -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>.")
endif()

project(CudaProject LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for, for example 75;86;89" FORCE)
endif()

set(CUDA_PTX_ARCH "sm_86" CACHE STRING "CUDA architecture used by the ptx-dump helper target")
set(NSYS_EXECUTABLE "" CACHE FILEPATH "Optional path to the Nsight Systems executable")

set(CUDA_KERNEL_SOURCES
    aes_tables.cu
    aes128_ctr.cu
    aes128_gcm.cu
    aes256_ctr.cu
    aes256_gcm.cu
    aes128_ecb.cu
    aes256_ecb.cu
)

set(CUDA_SOURCES
    ${CUDA_KERNEL_SOURCES}
    main.cu
)

add_executable(CudaProject ${CUDA_SOURCES})
add_executable(CudaAesKat ${CUDA_KERNEL_SOURCES} tests/kat_main.cu)

foreach(target_name IN ITEMS CudaProject CudaAesKat)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_definitions(${target_name} PRIVATE ENABLE_NVTX)

    target_link_libraries(${target_name} PRIVATE
        CUDA::cudart
        OpenSSL::SSL
        OpenSSL::Crypto
    )

    if(WIN32)
        target_link_libraries(${target_name} PRIVATE
            crypt32
            ws2_32
            user32
            gdi32
            advapi32
            kernel32
        )
    endif()

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo --ptxas-options=-v -O3>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2 /EHsc>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3>
    )
endforeach()

set_source_files_properties(${CUDA_KERNEL_SOURCES} main.cu tests/kat_main.cu PROPERTIES
    CUDA_MAXIMUM_REGISTER_COUNT 64
)

enable_testing()
add_test(NAME cuda_aes_kat COMMAND CudaAesKat)

add_custom_target(ptx-dump
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bench
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=${CUDA_PTX_ARCH} ${CMAKE_SOURCE_DIR}/aes128_ecb.cu -o ${CMAKE_BINARY_DIR}/bench/ptx_lookup.ptx
    COMMAND ${CMAKE_COMMAND} -E echo "PTX dump written to ${CMAKE_BINARY_DIR}/bench/ptx_lookup.ptx"
    COMMENT "Generating PTX for aes128_ecb.cu"
)

if(NSYS_EXECUTABLE)
    add_custom_target(nsight-profile
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bench
        COMMAND ${NSYS_EXECUTABLE} profile -o ${CMAKE_BINARY_DIR}/bench/my_run $<TARGET_FILE:CudaProject>
        DEPENDS CudaProject
        COMMENT "Profiling CudaProject with Nsight Systems"
    )
else()
    add_custom_target(nsight-profile
        COMMAND ${CMAKE_COMMAND} -E echo "Set -DNSYS_EXECUTABLE=<path-to-nsys> to enable Nsight Systems profiling."
        COMMENT "Nsight Systems executable was not configured"
    )
endif()

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <random>
#include <iostream>
#include "aes_common.h"

static void checkCuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t chunkBytes = 16 * (1 << 20); // 16 MiB per stream
    const size_t totalBytes = chunkBytes * 2;
    const size_t chunkBlocks = chunkBytes / 16;

    std::vector<uint8_t> h_plain(totalBytes);
    for (auto &b : h_plain) b = static_cast<uint8_t>(rand() & 0xFF);
    std::vector<uint8_t> h_cipher(totalBytes);

    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    init_T_tables();

    std::vector<uint8_t> key(16);
    for (auto &b : key) b = static_cast<uint8_t>(rand() & 0xFF);
    std::vector<uint32_t> roundKeys(44);
    expandKey128(key.data(), roundKeys.data());
    init_roundKeys(roundKeys.data(), roundKeys.size());

    uint8_t *d_plain[2] = {nullptr, nullptr};
    uint8_t *d_cipher[2] = {nullptr, nullptr};
    cudaStream_t streams[2];

    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaStreamCreate(&streams[i]), "cudaStreamCreate");
        checkCuda(cudaMalloc(&d_plain[i], chunkBytes), "cudaMalloc plain");
        checkCuda(cudaMalloc(&d_cipher[i], chunkBytes), "cudaMalloc cipher");
    }

    // Offsets for CTR counters
    uint64_t ctrLo0 = 0, ctrHi0 = 0;
    uint64_t ctrLo1 = chunkBlocks, ctrHi1 = 0;
    if (ctrLo1 < chunkBlocks) ctrHi1 = 1; // handle carry

    // Copy input chunks asynchronously
    checkCuda(cudaMemcpyAsync(d_plain[0], h_plain.data(), chunkBytes,
                              cudaMemcpyHostToDevice, streams[0]), "memcpyAsync H2D 0");
    checkCuda(cudaMemcpyAsync(d_plain[1], h_plain.data() + chunkBytes, chunkBytes,
                              cudaMemcpyHostToDevice, streams[1]), "memcpyAsync H2D 1");

    dim3 block(256);
    dim3 grid((unsigned)((chunkBlocks + block.x - 1) / block.x));

    aes128_ctr_encrypt<<<grid, block, 0, streams[0]>>>(d_plain[0], d_cipher[0], chunkBlocks,
                                                      ctrLo0, ctrHi0);
    aes128_ctr_encrypt<<<grid, block, 0, streams[1]>>>(d_plain[1], d_cipher[1], chunkBlocks,
                                                      ctrLo1, ctrHi1);

    checkCuda(cudaMemcpyAsync(h_cipher.data(), d_cipher[0], chunkBytes,
                              cudaMemcpyDeviceToHost, streams[0]), "memcpyAsync D2H 0");
    checkCuda(cudaMemcpyAsync(h_cipher.data() + chunkBytes, d_cipher[1], chunkBytes,
                              cudaMemcpyDeviceToHost, streams[1]), "memcpyAsync D2H 1");

    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaStreamSynchronize(streams[i]), "stream sync");
        cudaStreamDestroy(streams[i]);
        cudaFree(d_plain[i]);
        cudaFree(d_cipher[i]);
    }

    std::cout << "Encrypted " << totalBytes / (1<<20) << " MiB using two streams." << std::endl;
    return 0;
}


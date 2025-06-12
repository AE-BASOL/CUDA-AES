#include <cuda_runtime.h>
#include <curand.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "aes_common.h"

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void checkCurand(curandStatus_t st, const char *msg) {
    if (st != CURAND_STATUS_SUCCESS) {
        std::cerr << msg << " failed" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    std::string mode = "ecb";
    int keyBits = 128;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else if (std::strncmp(argv[i], "--key=", 6) == 0) {
            keyBits = std::atoi(argv[i] + 6);
        }
    }
    if (mode != "ecb" && mode != "ctr" && mode != "gcm") {
        std::cerr << "Invalid mode" << std::endl; return 1; }
    if (keyBits != 128 && keyBits != 256) {
        std::cerr << "Invalid key size" << std::endl; return 1; }

    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    init_T_tables();

    std::vector<uint8_t> key(keyBits/8, 0);
    std::vector<uint32_t> rk((keyBits==128)?44:60);
    if (keyBits==128) expandKey128(key.data(), rk.data());
    else expandKey256(key.data(), rk.data());
    init_roundKeys(rk.data(), (int)rk.size());

    const size_t dataBytes = 10 * (1<<20); // 10 MiB
    const size_t numBlocks = dataBytes / 16;

    uint8_t *d_plain=nullptr, *d_cipher=nullptr, *d_out=nullptr, *d_iv=nullptr, *d_tag=nullptr, *d_tag2=nullptr;
    checkCuda(cudaMalloc(&d_plain, dataBytes), "malloc plain");
    checkCuda(cudaMalloc(&d_cipher, dataBytes), "malloc cipher");
    checkCuda(cudaMalloc(&d_out, dataBytes), "malloc out");
    if (mode == "gcm") {
        checkCuda(cudaMalloc(&d_iv, 16), "malloc iv");
        checkCuda(cudaMalloc(&d_tag, 16), "malloc tag");
        checkCuda(cudaMalloc(&d_tag2, 16), "malloc tag2");
        uint8_t iv[16] = {0};
        checkCuda(cudaMemcpy(d_iv, iv, 16, cudaMemcpyHostToDevice), "copy iv");
    }

    curandGenerator_t gen;
    checkCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
    checkCurand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL), "curandSetSeed");
    checkCurand(curandGenerate(gen, reinterpret_cast<unsigned int*>(d_plain), dataBytes/4), "curandGenerate");
    curandDestroyGenerator(gen);

    std::vector<uint8_t> h_plain(dataBytes);
    std::vector<uint8_t> h_back(dataBytes);
    checkCuda(cudaMemcpy(h_plain.data(), d_plain, dataBytes, cudaMemcpyDeviceToHost), "copy plain back");

    dim3 block(256);
    dim3 grid((numBlocks + block.x - 1) / block.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (mode=="ecb" && keyBits==128)
        aes128_ecb_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks);
    else if (mode=="ecb" && keyBits==256)
        aes256_ecb_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks);
    else if (mode=="ctr" && keyBits==128)
        aes128_ctr_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks,0,0);
    else if (mode=="ctr" && keyBits==256)
        aes256_ctr_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks,0,0);
    else if (mode=="gcm" && keyBits==128)
        aes128_gcm_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks,d_iv,d_tag);
    else if (mode=="gcm" && keyBits==256)
        aes256_gcm_encrypt<<<grid,block>>>(d_plain,d_cipher,numBlocks,d_iv,d_tag);
    cudaEventRecord(stop);
    checkCuda(cudaGetLastError(), "encrypt kernel");
    cudaEventSynchronize(stop);

    float ms=0.0f;
    cudaEventElapsedTime(&ms,start,stop);

    if (mode=="ecb" && keyBits==128)
        aes128_ecb_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks);
    else if (mode=="ecb" && keyBits==256)
        aes256_ecb_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks);
    else if (mode=="ctr" && keyBits==128)
        aes128_ctr_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks,0,0);
    else if (mode=="ctr" && keyBits==256)
        aes256_ctr_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks,0,0);
    else if (mode=="gcm" && keyBits==128)
        aes128_gcm_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks,d_iv,d_tag,d_tag2);
    else if (mode=="gcm" && keyBits==256)
        aes256_gcm_decrypt<<<grid,block>>>(d_cipher,d_out,numBlocks,d_iv,d_tag,d_tag2);
    checkCuda(cudaGetLastError(), "decrypt kernel");

    checkCuda(cudaMemcpy(h_back.data(), d_out, dataBytes, cudaMemcpyDeviceToHost), "copy back");

    bool ok = std::memcmp(h_plain.data(), h_back.data(), dataBytes) == 0;
    double throughput = (double)dataBytes / (1<<30) / (ms/1e3);
    std::cout.setf(std::ios::fixed);
    std::cout<<std::setprecision(5);
    std::cout << "[GPU] " << mode << "-" << keyBits << " processed "
              << (dataBytes>>20) << " MiB in " << ms << " ms -> ";
    std::cout<<std::setprecision(4)<<throughput<<" GiB/s"<<std::endl;
    std::cout << "Round-trip " << mode << "-" << keyBits << " "
              << (dataBytes>>20) << " MiB " << (ok?"PASS":"FAIL") << std::endl;

    cudaFree(d_plain); cudaFree(d_cipher); cudaFree(d_out);
    if (d_iv) { cudaFree(d_iv); cudaFree(d_tag); cudaFree(d_tag2); }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ok?0:1;
}

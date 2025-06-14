#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "aes_common.h"
#include "profiling_helpers.h"

// Error checking helper
#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Pack 96-bit IV into ctrLo/ctrHi expected by CTR kernels
static void packCtr(const uint8_t iv[12], uint64_t &ctrLo, uint64_t &ctrHi) {
    uint32_t w0=0,w1=0,w2=0;
    memcpy(&w0, iv, 4);
    memcpy(&w1, iv+4,4);
    memcpy(&w2, iv+8,4);
    uint32_t w3 = 0x01000000u; // counter=1 big-endian
    ctrLo = (uint64_t)w0 | ((uint64_t)w1<<32);
    ctrHi = (uint64_t)w2 | ((uint64_t)w3<<32);
}

static void printKeySchedule(const uint32_t *keys, size_t numWords, const char *label) {
    std::cout << label << ": ";
    for (size_t i = 0; i < numWords; ++i) {
        std::cout << std::hex << std::setfill('0') << std::setw(8) << keys[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

static void printInputData(const uint8_t *data, size_t len, const char *label) {
    std::cout << label << ": ";
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)data[i] << " ";
    }
    std::cout << std::dec << std::endl;
}

int main() {
#ifdef ENABLE_NVTX
    printf("ENABLE_NVTX is defined!\n");
#else
    printf("ENABLE_NVTX is NOT defined!\n");
#endif
    init_T_tables();
    std::vector<size_t> testSizes = {10 << 20};
    std::vector<std::string> modes = {"ecb", "ctr", "gcm"};
    std::vector<int> keyBitsOptions = {128, 256};

    std::mt19937_64 rng(12345ULL);
    for (const std::string &mode : modes) {
        for (int keyBits : keyBitsOptions) {
            size_t keyBytes = keyBits / 8;
            std::vector<uint8_t> key(keyBytes);
            for (auto &b : key) b = uint8_t(rng() & 0xFF);
            std::vector<uint8_t> iv(16, 0);
            if (mode != "ecb")
                for (size_t i=0;i<12;++i) iv[i] = uint8_t(rng() & 0xFF);

            std::vector<uint32_t> roundKeys(keyBits==128?44:60);
            if (keyBits==128) expandKey128(key.data(), roundKeys.data());
            else expandKey256(key.data(), roundKeys.data());
            printKeySchedule(roundKeys.data(), roundKeys.size(), "Host Key Schedule");
            init_roundKeys(roundKeys.data(), (int)roundKeys.size());
            std::vector<uint32_t> deviceKeys(roundKeys.size());
            // NVTX Start: Memcpy Device→Host
            NVTX_PUSH("Memcpy Device\xE2\x86\x92Host");
            CHECK_CUDA(cudaMemcpyFromSymbol(deviceKeys.data(), d_roundKeys, roundKeys.size()*sizeof(uint32_t)));
            NVTX_POP();
            // NVTX End: Memcpy Device→Host
            printKeySchedule(deviceKeys.data(), deviceKeys.size(), "Device Key Schedule");

            for (size_t dataBytes : testSizes) {
                size_t nBlocks = (dataBytes + 15) / 16;
                dataBytes = nBlocks * 16;
                std::vector<uint8_t> h_plain(dataBytes);
                for (auto &b : h_plain) b = uint8_t(rng() & 0xFF);
                std::vector<uint8_t> h_cipher(dataBytes);
                std::vector<uint8_t> h_recovered(dataBytes);
                uint8_t tagCPU[16], tagGPU[16];

                uint8_t *d_plain=nullptr,*d_cipher=nullptr,*d_tag=nullptr;
                // NVTX Start: Malloc
                NVTX_PUSH("Malloc");
                CHECK_CUDA(cudaMalloc(&d_plain, dataBytes));
                NVTX_POP();
                // NVTX End: Malloc
                // NVTX Start: Malloc
                NVTX_PUSH("Malloc");
                CHECK_CUDA(cudaMalloc(&d_cipher, dataBytes));
                NVTX_POP();
                // NVTX End: Malloc
                if (mode=="gcm") {
                    // NVTX Start: Malloc
                    NVTX_PUSH("Malloc");
                    CHECK_CUDA(cudaMalloc(&d_tag,16));
                    NVTX_POP();
                    // NVTX End: Malloc
                }
                printInputData(h_plain.data(), std::min<size_t>(64, h_plain.size()), "Host Input Data");
                // NVTX Start: Memcpy Host→Device
                NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                CHECK_CUDA(cudaMemcpy(d_plain, h_plain.data(), dataBytes, cudaMemcpyHostToDevice));
                NVTX_POP();
                // NVTX End: Memcpy Host→Device
                std::vector<uint8_t> deviceInput(h_plain.size());
                // NVTX Start: Memcpy Device→Host
                NVTX_PUSH("Memcpy Device\xE2\x86\x92Host");
                CHECK_CUDA(cudaMemcpy(deviceInput.data(), d_plain, dataBytes, cudaMemcpyDeviceToHost));
                NVTX_POP();
                // NVTX End: Memcpy Device→Host
                printInputData(deviceInput.data(), std::min<size_t>(64, deviceInput.size()), "Device Input Data");

                dim3 block(256);
                dim3 grid((unsigned)((nBlocks + block.x - 1)/block.x));
                cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
                cudaEventRecord(start);
                if (mode=="ecb" && keyBits==128) {
                    // NVTX Start: ECB-128 Encrypt
                    NVTX_PUSH("ECB-128 Encrypt");
                    aes128_ecb_encrypt<<<grid,block>>>(d_plain,d_cipher,nBlocks);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: ECB-128 Encrypt
                } else if (mode=="ecb" && keyBits==256) {
                    // NVTX Start: ECB-256 Encrypt
                    NVTX_PUSH("ECB-256 Encrypt");
                    aes256_ecb_encrypt<<<grid,block>>>(d_plain,d_cipher,nBlocks);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: ECB-256 Encrypt
                } else if (mode=="ctr" && keyBits==128) {
                    uint64_t ctrLo=0, ctrHi=0; packCtr(iv.data(),ctrLo,ctrHi);
                    // NVTX Start: CTR-128 Encrypt
                    NVTX_PUSH("CTR-128 Encrypt");
                    aes128_ctr_encrypt<<<grid,block>>>(d_plain,d_cipher,nBlocks,ctrLo,ctrHi);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: CTR-128 Encrypt
                } else if (mode=="ctr" && keyBits==256) {
                    uint64_t ctrLo=0, ctrHi=0; packCtr(iv.data(),ctrLo,ctrHi);
                    // NVTX Start: CTR-256 Encrypt
                    NVTX_PUSH("CTR-256 Encrypt");
                    aes256_ctr_encrypt<<<grid,block>>>(d_plain,d_cipher,nBlocks,ctrLo,ctrHi);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: CTR-256 Encrypt
                } else if (mode=="gcm" && keyBits==128) {
                    uint8_t *d_iv=nullptr; CHECK_CUDA(cudaMalloc(&d_iv,12));
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(),12,cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: GCM-128 Encrypt
                    NVTX_PUSH("GCM-128 Encrypt");
                    aes128_gcm_encrypt<<<1,256>>>(d_plain,d_cipher,nBlocks,d_iv,d_tag);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: GCM-128 Encrypt
                    // NVTX Start: Free
                    NVTX_PUSH("Free");
                    CHECK_CUDA(cudaFree(d_iv));
                    NVTX_POP();
                    // NVTX End: Free
                } else if (mode=="gcm" && keyBits==256) {
                    uint8_t *d_iv=nullptr; CHECK_CUDA(cudaMalloc(&d_iv,12));
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(),12,cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: GCM-256 Encrypt
                    NVTX_PUSH("GCM-256 Encrypt");
                    aes256_gcm_encrypt<<<1,256>>>(d_plain,d_cipher,nBlocks,d_iv,d_tag);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: GCM-256 Encrypt
                    // NVTX Start: Free
                    NVTX_PUSH("Free");
                    CHECK_CUDA(cudaFree(d_iv));
                    NVTX_POP();
                    // NVTX End: Free
                }
                cudaEventRecord(stop);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaEventSynchronize(stop));
                float ms=0.0f; cudaEventElapsedTime(&ms,start,stop);
                double gb = (double)dataBytes/(1<<30); double thr = gb/(ms/1000.0);
                std::cout << "[GPU] " << mode << "-" << keyBits << " processed "
                          << (double)dataBytes/(1<<20) << " MiB in " << ms
                          << " ms -> " << thr << " GiB/s" << std::endl;

                // NVTX Start: Memcpy Device→Host
                NVTX_PUSH("Memcpy Device\xE2\x86\x92Host");
                CHECK_CUDA(cudaMemcpy(h_cipher.data(), d_cipher, dataBytes, cudaMemcpyDeviceToHost));
                NVTX_POP();
                // NVTX End: Memcpy Device→Host
                if (mode=="gcm") {
                    // NVTX Start: Memcpy Device→Host
                    NVTX_PUSH("Memcpy Device\xE2\x86\x92Host");
                    CHECK_CUDA(cudaMemcpy(tagGPU, d_tag, 16, cudaMemcpyDeviceToHost));
                    NVTX_POP();
                    // NVTX End: Memcpy Device→Host
                }

                if (mode=="ecb" && keyBits==128) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: ECB-128 Decrypt
                    NVTX_PUSH("ECB-128 Decrypt");
                    aes128_ecb_decrypt<<<grid,block>>>(d_cipher,d_plain,nBlocks);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: ECB-128 Decrypt
                } else if (mode=="ecb" && keyBits==256) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: ECB-256 Decrypt
                    NVTX_PUSH("ECB-256 Decrypt");
                    aes256_ecb_decrypt<<<grid,block>>>(d_cipher,d_plain,nBlocks);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: ECB-256 Decrypt
                } else if (mode=="ctr" && keyBits==128) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    uint64_t ctrLo=0, ctrHi=0; packCtr(iv.data(),ctrLo,ctrHi);
                    // NVTX Start: CTR-128 Decrypt
                    NVTX_PUSH("CTR-128 Decrypt");
                    aes128_ctr_decrypt<<<grid,block>>>(d_cipher,d_plain,nBlocks,ctrLo,ctrHi);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: CTR-128 Decrypt
                } else if (mode=="ctr" && keyBits==256) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    uint64_t ctrLo=0, ctrHi=0; packCtr(iv.data(),ctrLo,ctrHi);
                    // NVTX Start: CTR-256 Decrypt
                    NVTX_PUSH("CTR-256 Decrypt");
                    aes256_ctr_decrypt<<<grid,block>>>(d_cipher,d_plain,nBlocks,ctrLo,ctrHi);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: CTR-256 Decrypt
                } else if (mode=="gcm" && keyBits==128) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    uint8_t *d_iv2=nullptr; CHECK_CUDA(cudaMalloc(&d_iv2,12));
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_iv2, iv.data(),12,cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: GCM-128 Decrypt
                    NVTX_PUSH("GCM-128 Decrypt");
                    aes128_gcm_decrypt<<<1,256>>>(d_cipher,d_plain,nBlocks,d_iv2,d_tag,d_tag);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: GCM-128 Decrypt
                    // NVTX Start: Free
                    NVTX_PUSH("Free");
                    CHECK_CUDA(cudaFree(d_iv2));
                    NVTX_POP();
                    // NVTX End: Free
                } else if (mode=="gcm" && keyBits==256) {
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    uint8_t *d_iv2=nullptr; CHECK_CUDA(cudaMalloc(&d_iv2,12));
                    // NVTX Start: Memcpy Host→Device
                    NVTX_PUSH("Memcpy Host\xE2\x86\x92Device");
                    CHECK_CUDA(cudaMemcpy(d_iv2, iv.data(),12,cudaMemcpyHostToDevice));
                    NVTX_POP();
                    // NVTX End: Memcpy Host→Device
                    // NVTX Start: GCM-256 Decrypt
                    NVTX_PUSH("GCM-256 Decrypt");
                    aes256_gcm_decrypt<<<1,256>>>(d_cipher,d_plain,nBlocks,d_iv2,d_tag,d_tag);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    NVTX_POP();
                    // NVTX End: GCM-256 Decrypt
                    // NVTX Start: Free
                    NVTX_PUSH("Free");
                    CHECK_CUDA(cudaFree(d_iv2));
                    NVTX_POP();
                    // NVTX End: Free
                }
                CHECK_CUDA(cudaDeviceSynchronize());
                // NVTX Start: Memcpy Device→Host
                NVTX_PUSH("Memcpy Device\xE2\x86\x92Host");
                CHECK_CUDA(cudaMemcpy(h_recovered.data(), d_plain, dataBytes, cudaMemcpyDeviceToHost));
                NVTX_POP();
                // NVTX End: Memcpy Device→Host
                bool match = (h_recovered == h_plain);
                std::cout << "    Round-trip " << mode << "-" << keyBits << " "
                          << (double)dataBytes/(1<<20) << " MiB "
                          << (match?"PASS":"FAIL") << std::endl;
                if(!match) {
                    for(size_t i=0;i<std::min<size_t>(64,dataBytes);++i) {
                        if(h_recovered[i]!=h_plain[i]) {
                            printf("Mismatch at byte %zu: got %02x, exp %02x\n", i, h_recovered[i], h_plain[i]);
                            break;
                        }
                    }
                }
#ifdef ENABLE_OPENSSL_VALIDATION
                {
                    const EVP_CIPHER *cipher_algo=nullptr;
                    if (mode=="ecb" && keyBits==128) cipher_algo = EVP_aes_128_ecb();
                    else if (mode=="ecb" && keyBits==256) cipher_algo = EVP_aes_256_ecb();
                    else if (mode=="ctr" && keyBits==128) cipher_algo = EVP_aes_128_ctr();
                    else if (mode=="ctr" && keyBits==256) cipher_algo = EVP_aes_256_ctr();
                    else if (mode=="gcm" && keyBits==128) cipher_algo = EVP_aes_128_gcm();
                    else if (mode=="gcm" && keyBits==256) cipher_algo = EVP_aes_256_gcm();
                    if (!cipher_algo) {
                        std::cerr << "[CPU] OpenSSL cipher not available for " << mode << "-" << keyBits << "\n";
                    } else {
                        std::vector<uint8_t> cpu_out(dataBytes);
                        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
                        EVP_EncryptInit_ex(ctx, cipher_algo, nullptr, nullptr, nullptr);
                        EVP_CIPHER_CTX_set_padding(ctx,0);
                        EVP_EncryptInit_ex(ctx,nullptr,nullptr,key.data(),(mode=="ecb"?nullptr:iv.data()));
                        int outLen=0,totalLen=0;
                        EVP_EncryptUpdate(ctx,cpu_out.data(),&outLen,h_plain.data(),(int)dataBytes);
                        totalLen += outLen;
                        EVP_EncryptFinal_ex(ctx,cpu_out.data()+totalLen,&outLen);
                        totalLen += outLen;
                        if (mode=="gcm") {
                            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tagCPU);
                        }
                        EVP_CIPHER_CTX_free(ctx);
                        bool ok=true;
                        if (mode=="gcm") {
                            ok = (cpu_out==h_cipher) && (std::memcmp(tagCPU,tagGPU,16)==0);
                        } else {
                            ok = (cpu_out==h_cipher);
                        }
                        std::cout << "    OpenSSL CPU match: " << (ok?"[✓]":"[✗]") << std::endl;
                    }
                }
#endif
                if(d_tag) cudaMemset(d_tag,0,16);
                // NVTX Start: Free
                NVTX_PUSH("Free");
                cudaFree(d_plain); cudaFree(d_cipher); if(d_tag) cudaFree(d_tag);
                NVTX_POP();
                // NVTX End: Free
            }
        }
    }
    return 0;
}

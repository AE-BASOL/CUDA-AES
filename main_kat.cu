#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "aes_common.h"
#define ENABLE_NVTX
#include "profiling_helpers.h"


// Simple error-check macro for CUDA calls
#define CHECK_CUDA(err)  do { \
cudaError_t _err = (err); \
if (_err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
exit(EXIT_FAILURE); \
} \
} while(0)

// Helper to print a buffer as hex
static void printHex(const char *label, const uint8_t *buf, size_t len) {
    printf("%s:", label);
    for (size_t i = 0; i < len; ++i) printf(" %02x", buf[i]);
    printf("\n");
}

// Helper to pack a 96-bit IV into the ctrLo/ctrHi format expected by the
// little-endian CTR kernels.  The IV is interpreted in network byte order and
// the 32-bit counter is initialised to 1.
static void packCtr(const uint8_t iv[12], uint64_t &ctrLo, uint64_t &ctrHi) {
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    memcpy(&w0, iv, 4);
    memcpy(&w1, iv + 4, 4);
    memcpy(&w2, iv + 8, 4);
    uint32_t w3 = 0x01000000u; // counter = 1 in big-endian
    ctrLo = (uint64_t)w0 | ((uint64_t)w1 << 32);
    ctrHi = (uint64_t)w2 | ((uint64_t)w3 << 32);
}

int main() {
    // Initialize tables in device constant memory
    init_T_tables();

    // Benchmark: measure throughput for each mode/keysize at various message lengths
    std::vector<size_t> testSizes = {10 << 20}; //{1 << 20, 10 << 20, 100 << 20, 1000 << 20}; // 1MB, 10MB, 100MB, 1000MB
    std::vector<std::string> modes = {"ecb", "ctr", "gcm"};
    std::vector<int> keyBitsOptions = {128, 256};

    std::mt19937_64 rng(12345ULL); // fixed seed for reproducibility
    for (const std::string &mode: modes) {
        for (int keyBits: keyBitsOptions) {
            // Prepare random key and IV
            size_t keyBytes = keyBits / 8;
            std::vector<uint8_t> key(keyBytes);
            for (auto &b: key) b = uint8_t(rng() & 0xFF);
            std::vector<uint8_t> iv(16, 0); // 16-byte IV buffer (for CTR/GCM: only first 12 bytes used, rest 0)
            if (mode != "ecb") {
                for (size_t i = 0; i < 12; ++i) iv[i] = uint8_t(rng() & 0xFF);
            }
            // Expand key and set constant memory
            std::vector<uint32_t> roundKeys((keyBits == 128) ? 44 : 60);
            if (keyBits == 128) expandKey128(key.data(), roundKeys.data());
            else expandKey256(key.data(), roundKeys.data());
            init_roundKeys(roundKeys.data(), (int) roundKeys.size());

            for (size_t dataBytes: testSizes) {
                size_t nBlocks = (dataBytes + 15) / 16;
                dataBytes = nBlocks * 16; // round up to full blocks
                // Allocate and fill host plaintext
                std::vector<uint8_t> h_plain(dataBytes);
                for (auto &b: h_plain) b = uint8_t(rng() & 0xFF);
                std::vector<uint8_t> h_cipher(dataBytes);
                std::vector<uint8_t> h_recovered(dataBytes);
                uint8_t tagCPU[16], tagGPU[16];
                // Allocate device buffers
                uint8_t *d_plain = nullptr, *d_cipher = nullptr, *d_tag = nullptr;
                CHECK_CUDA(cudaMalloc(&d_plain, dataBytes));
                CHECK_CUDA(cudaMalloc(&d_cipher, dataBytes));
                if (mode == "gcm") {
                    CHECK_CUDA(cudaMalloc(&d_tag, 16));
                }
                // Copy plaintext to device
                CHECK_CUDA(cudaMemcpy(d_plain, h_plain.data(), dataBytes, cudaMemcpyHostToDevice));
                // Configure launch (256 threads per block for large data)
                dim3 block(256);
                dim3 grid((unsigned) ((nBlocks + block.x - 1) / block.x));
                // Time the encryption kernel
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                if (mode == "ecb" && keyBits == 128) {
                    aes128_ecb_encrypt<<<grid, block>>>(d_plain, d_cipher, nBlocks);
                } else if (mode == "ecb" && keyBits == 256) {
                    aes256_ecb_encrypt<<<grid, block>>>(d_plain, d_cipher, nBlocks);
                } else if (mode == "ctr" && keyBits == 128) {
                    uint64_t ctrHi = 0, ctrLo = 0;
                    packCtr(iv.data(), ctrLo, ctrHi);
                    aes128_ctr_encrypt<<<grid, block>>>(d_plain, d_cipher, nBlocks, ctrLo, ctrHi);
                } else if (mode == "ctr" && keyBits == 256) {
                    uint64_t ctrHi = 0, ctrLo = 0;
                    packCtr(iv.data(), ctrLo, ctrHi);
                    aes256_ctr_encrypt<<<grid, block>>>(d_plain, d_cipher, nBlocks, ctrLo, ctrHi);
                } else if (mode == "gcm" && keyBits == 128) {
                    // Copy IV to device (12 bytes)
                    uint8_t *d_iv = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_iv, 12));
                    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), 12, cudaMemcpyHostToDevice));
                    aes128_gcm_encrypt<<<1, 256>>>(d_plain, d_cipher, nBlocks, d_iv, d_tag);
                    CHECK_CUDA(cudaFree(d_iv));
                } else if (mode == "gcm" && keyBits == 256) {
                    uint8_t *d_iv = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_iv, 12));
                    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), 12, cudaMemcpyHostToDevice));
                    aes256_gcm_encrypt<<<1, 256>>>(d_plain, d_cipher, nBlocks, d_iv, d_tag);
                    CHECK_CUDA(cudaFree(d_iv));
                }
                cudaEventRecord(stop);
                CHECK_CUDA(cudaGetLastError()); // check launch
                CHECK_CUDA(cudaEventSynchronize(stop));
                // Compute throughput
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, start, stop);
                double gb = (double) dataBytes / (1 << 30);
                double throughput = gb / (ms / 1000.0);
                std::cout << "[GPU] " << mode << "-" << keyBits << " processed "
                        << (double) dataBytes / (1 << 20) << " MiB in " << ms << " ms -> "
                        << throughput << " GiB/s" << std::endl;

                // Copy ciphertext (and tag if GCM) back to host
                CHECK_CUDA(cudaMemcpy(h_cipher.data(), d_cipher, dataBytes, cudaMemcpyDeviceToHost));
                if (mode == "gcm") {
                    CHECK_CUDA(cudaMemcpy(tagGPU, d_tag, 16, cudaMemcpyDeviceToHost));
                }
                // Launch decryption and verify round-trip
                if (mode == "ecb" && keyBits == 128) {
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    aes128_ecb_decrypt<<<grid, block>>>(d_cipher, d_plain, nBlocks);
                } else if (mode == "ecb" && keyBits == 256) {
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    aes256_ecb_decrypt<<<grid, block>>>(d_cipher, d_plain, nBlocks);
                } else if (mode == "ctr" && keyBits == 128) {
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    uint64_t ctrHi = 0, ctrLo = 0;
                    packCtr(iv.data(), ctrLo, ctrHi);
                    aes128_ctr_decrypt<<<grid, block>>>(d_cipher, d_plain, nBlocks, ctrLo, ctrHi);
                } else if (mode == "ctr" && keyBits == 256) {
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    uint64_t ctrHi = 0, ctrLo = 0;
                    packCtr(iv.data(), ctrLo, ctrHi);
                    aes256_ctr_decrypt<<<grid, block>>>(d_cipher, d_plain, nBlocks, ctrLo, ctrHi);
                } else if (mode == "gcm" && keyBits == 128) {
                    // Copy cipher back to device and decrypt (generate tag as well)
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    uint8_t *d_iv2 = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_iv2, 12));
                    CHECK_CUDA(cudaMemcpy(d_iv2, iv.data(), 12, cudaMemcpyHostToDevice));
                    // Use provided GPU tag output as expected tag input (simulate verification)
                    aes128_gcm_decrypt<<<1, 256>>>(d_cipher, d_plain, nBlocks, d_iv2, d_tag, d_tag);
                    CHECK_CUDA(cudaFree(d_iv2));
                } else if (mode == "gcm" && keyBits == 256) {
                    CHECK_CUDA(cudaMemcpy(d_cipher, h_cipher.data(), dataBytes, cudaMemcpyHostToDevice));
                    uint8_t *d_iv2 = nullptr;
                    CHECK_CUDA(cudaMalloc(&d_iv2, 12));
                    CHECK_CUDA(cudaMemcpy(d_iv2, iv.data(), 12, cudaMemcpyHostToDevice));
                    aes256_gcm_decrypt<<<1, 256>>>(d_cipher, d_plain, nBlocks, d_iv2, d_tag, d_tag);
                    CHECK_CUDA(cudaFree(d_iv2));
                }
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_recovered.data(), d_plain, dataBytes, cudaMemcpyDeviceToHost));
                bool match = (h_recovered == h_plain);
                std::cout << "    Round-trip " << mode << "-" << keyBits << " "
                        << (double) dataBytes / (1 << 20) << " MiB "
                        << (match ? "PASS" : "FAIL") << std::endl;
                if (!match) {
                    // If data is large, just print first few bytes difference
                    for (size_t i = 0; i < std::min<size_t>(64, dataBytes); ++i) {
                        if (h_recovered[i] != h_plain[i]) {
                            printf("Mismatch at byte %zu: got %02x, exp %02x\n", i, h_recovered[i], h_plain[i]);
                            break;
                        }
                    }
                }
#ifdef ENABLE_OPENSSL_VALIDATION
                // CPU validation using OpenSSL EVP (if enabled)
                {
                    const EVP_CIPHER *cipher_algo = nullptr;
                    if (mode == "ecb" && keyBits == 128) cipher_algo = EVP_aes_128_ecb();
                    else if (mode == "ecb" && keyBits == 256) cipher_algo = EVP_aes_256_ecb();
                    else if (mode == "ctr" && keyBits == 128) cipher_algo = EVP_aes_128_ctr();
                    else if (mode == "ctr" && keyBits == 256) cipher_algo = EVP_aes_256_ctr();
                    else if (mode == "gcm" && keyBits == 128) cipher_algo = EVP_aes_128_gcm();
                    else if (mode == "gcm" && keyBits == 256) cipher_algo = EVP_aes_256_gcm();
                    if (!cipher_algo) {
                        std::cerr << "[CPU] OpenSSL cipher not available for " << mode << "-" << keyBits << "\n";
                    } else {
                        std::vector<uint8_t> cpu_out(dataBytes);
                        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
                        EVP_EncryptInit_ex(ctx, cipher_algo, nullptr, nullptr, nullptr);
                        EVP_CIPHER_CTX_set_padding(ctx, 0);
                        EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), (mode=="ecb" ? nullptr : iv.data()));
                        int outLen = 0, totalLen = 0;
                        EVP_EncryptUpdate(ctx, cpu_out.data(), &outLen, h_plain.data(), (int)dataBytes);
                        totalLen += outLen;
                        EVP_EncryptFinal_ex(ctx, cpu_out.data()+totalLen, &outLen);
                        totalLen += outLen;
                        if (mode == "gcm") {
                            // Get GCM tag
                            EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tagCPU);
                        }
                        EVP_CIPHER_CTX_free(ctx);
                        bool ok = true;
                        if (mode == "gcm") {
                            ok = (cpu_out == h_cipher) && (std::memcmp(tagCPU, tagGPU, 16) == 0);
                        } else {
                            ok = (cpu_out == h_cipher);
                        }
                        std::cout << "    OpenSSL CPU match: " << (ok ? "[✓]" : "[✗]") << std::endl;
                    }
                }
#endif
                // Cleanup device buffers for this size
                if (d_tag) cudaMemset(d_tag, 0, 16); // clear tag buffer for reuse
            }
            // Free device buffers for this mode/key
            cudaFree(d_plain);
            cudaFree(d_cipher);
            if (d_tag) cudaFree(d_tag);
        }
    }
    return 0;
}

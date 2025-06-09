#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "aes_common.h"
#define ENABLE_NVTX     // ❶ derleme zamanı anahtarı
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

// Dump expanded round keys for debugging
static void dumpRoundKeys(const char *label, const uint32_t *rk, size_t words) {
    printf("%s (\%zu words)\n", label, words);
    for (size_t i = 0; i < words; ++i) {
        if (i % 4 == 0) printf("  %2zu:", i);
        printf(" %08x", rk[i]);
        if (i % 4 == 3) printf("\n");
    }
    if (words % 4) printf("\n");
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
    uint8_t *d_plain = nullptr, *d_cipher = nullptr, *d_tag = nullptr;
    // Known-Answer Tests (KAT) for functional verification
    bool allPassed = true;
    printf("Running AES Known-Answer Tests...\n");

    // ---------- Test 1: AES-128-ECB single block ----------
    NVTX_PUSH("KAT_AES128_ECB"); {
        const char *testName = "AES-128-ECB";
        uint8_t key128[16] = {
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
        };
        uint8_t plaintext[16] = {
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
        };
        uint8_t expectedCipher[16] = {
            0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32
        };
        // Expand key and upload
        std::vector<uint32_t> roundKeys(44);
        expandKey128(key128, roundKeys.data());
        dumpRoundKeys("AES-128 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        // Allocate device buffers
        uint8_t *d_in = nullptr, *d_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_in, 16));
        CHECK_CUDA(cudaMalloc(&d_out, 16));
        CHECK_CUDA(cudaMemcpy(d_in, plaintext, 16, cudaMemcpyHostToDevice));
        // Launch encryption and copy result
        aes128_ecb_encrypt<<<1,1>>>(d_in, d_out, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_out, 16, cudaMemcpyDeviceToHost));
        bool pass = (std::memcmp(cipher, expectedCipher, 16) == 0);
        printf("[TEST] %s encryption : %s\n", testName, pass ? "PASS" : "FAIL");
        if (!pass) {
            printHex("  Exp", expectedCipher, 16);
            printHex("  Got", cipher, 16);
        }
        allPassed &= pass;
        // Now test decryption (round-trip check)
        uint8_t result[16] = {0};
        CHECK_CUDA(cudaMemcpy(d_in, cipher, 16, cudaMemcpyHostToDevice));
        aes128_ecb_decrypt<<<1,1>>>(d_in, d_out, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(result, d_out, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(result, plaintext, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", result, 16);
        }
        allPassed &= decPass;
        cudaFree(d_in);
        cudaFree(d_out);
    }
    NVTX_POP();

    // ---------- Test 2: AES-256-ECB single block ----------
    NVTX_PUSH("KAT_AES256_ECB");
    {
        const char *testName = "AES-256-ECB";
        uint8_t key256[32] = {
            0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
            0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
            0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
            0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4
        };
        uint8_t plaintext[16] = {
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a
        };
        uint8_t expectedCipher[16] = {
            0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0x36, 0x3e,
            0x12, 0x20, 0x34, 0x12, 0x3c, 0xd6, 0x5a, 0x42
        };
        std::vector<uint32_t> roundKeys(60);
        expandKey256(key256, roundKeys.data());
        dumpRoundKeys("AES-256 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        uint8_t *d_in = nullptr, *d_out = nullptr;
        CHECK_CUDA(cudaMalloc(&d_in, 16));
        CHECK_CUDA(cudaMalloc(&d_out, 16));
        CHECK_CUDA(cudaMemcpy(d_in, plaintext, 16, cudaMemcpyHostToDevice));
        aes256_ecb_encrypt<<<1,1>>>(d_in, d_out, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_out, 16, cudaMemcpyDeviceToHost));
        bool pass = (std::memcmp(cipher, expectedCipher, 16) == 0);
        printf("[TEST] %s encryption : %s\n", testName, pass ? "PASS" : "FAIL");
        if (!pass) {
            printHex("  Exp", expectedCipher, 16);
            printHex("  Got", cipher, 16);
        }
        allPassed &= pass;
        // Decrypt round-trip
        uint8_t result[16];
        CHECK_CUDA(cudaMemcpy(d_in, cipher, 16, cudaMemcpyHostToDevice));
        aes256_ecb_decrypt<<<1,1>>>(d_in, d_out, 1);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(result, d_out, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(result, plaintext, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", result, 16);
        }
        allPassed &= decPass;
        cudaFree(d_in);
        cudaFree(d_out);
    }
    NVTX_POP();

    // ---------- Test 3: AES-128-CTR (RFC 3686 test vector) ----------
        NVTX_PUSH("KAT_AES128_CTR");
    {
        const char *testName = "AES-128-CTR";
        uint8_t key[16] = {
            0xAE, 0x68, 0x52, 0xF8, 0x12, 0x10, 0x67, 0xCC,
            0x4B, 0xF7, 0xA5, 0x76, 0x55, 0x77, 0xF3, 0x9E
        };
        uint8_t iv[12] = {
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30
        };
        uint8_t plaintext[16] = {
            0x53, 0x69, 0x6E, 0x67, 0x6C, 0x65, 0x20, 0x62,
            0x6C, 0x6F, 0x63, 0x6B, 0x20, 0x6D, 0x73, 0x67
        }; // "Single block msg"
        uint8_t expectedCipher[16] = {
            0xE4, 0x09, 0x5D, 0x4F, 0xB7, 0xA7, 0xB3, 0x79,
            0x2D, 0x61, 0x75, 0xA3, 0x26, 0x13, 0x11, 0xB8
        };
        std::vector<uint32_t> roundKeys(44);
        expandKey128(key, roundKeys.data());
        dumpRoundKeys("AES-128 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        uint8_t *d_plain = nullptr, *d_cipher = nullptr;
        CHECK_CUDA(cudaMalloc(&d_plain, 16));
        CHECK_CUDA(cudaMalloc(&d_cipher, 16));
        CHECK_CUDA(cudaMemcpy(d_plain, plaintext, 16, cudaMemcpyHostToDevice));
        // Prepare 128-bit IV|counter for little-endian kernel
        uint64_t ctrHi = 0, ctrLo = 0;
        packCtr(iv, ctrLo, ctrHi);
        aes128_ctr_encrypt<<<1,1>>>(d_plain, d_cipher, 1, ctrLo, ctrHi);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_cipher, 16, cudaMemcpyDeviceToHost));
        bool pass = (std::memcmp(cipher, expectedCipher, 16) == 0);
        printf("[TEST] %s encryption : %s\n", testName, pass ? "PASS" : "FAIL");
        if (!pass) {
            printHex("  Exp", expectedCipher, 16);
            printHex("  Got", cipher, 16);
        }
        allPassed &= pass;
        // Round-trip decryption
        uint8_t result[16];
        CHECK_CUDA(cudaMemcpy(d_plain, cipher, 16, cudaMemcpyHostToDevice));
        aes128_ctr_decrypt<<<1,1>>>(d_plain, d_cipher, 1, ctrLo, ctrHi);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(result, d_cipher, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(result, plaintext, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", result, 16);
        }
        allPassed &= decPass;
        cudaFree(d_plain);
        cudaFree(d_cipher);
    }
    NVTX_POP();

    // ---------- Test 4: AES-256-CTR (RFC 3686 test vector) ----------
    NVTX_PUSH("KAT_AES256_CTR");
    {
        const char *testName = "AES-256-CTR";
        uint8_t key[32] = {
            0x77, 0x6B, 0xEF, 0xF2, 0x85, 0x1D, 0xB0, 0x6F,
            0x4C, 0x8A, 0x05, 0x42, 0xC8, 0x69, 0x6F, 0x6C,
            0x6A, 0x81, 0xAF, 0x1E, 0xEC, 0x96, 0xB4, 0xD3,
            0x7F, 0xC1, 0xD6, 0x89, 0xE6, 0xC1, 0xC1, 0x04
        };
        uint8_t iv[12] = {
            0xDB, 0x56, 0x72, 0xC9, 0x7A, 0xA8, 0xF0, 0xB2, 0x00, 0x00, 0x00, 0x60
        };
        uint8_t plaintext[16] = {
            0x53, 0x69, 0x6E, 0x67, 0x6C, 0x65, 0x20, 0x62,
            0x6C, 0x6F, 0x63, 0x6B, 0x20, 0x6D, 0x73, 0x67
        };
        uint8_t expectedCipher[16] = {
            0x14, 0x5A, 0xD0, 0x1D, 0xBF, 0x82, 0x4E, 0xC7,
            0x56, 0x08, 0x63, 0xDC, 0x71, 0xE3, 0xE0, 0xC0
        };
        std::vector<uint32_t> roundKeys(60);
        expandKey256(key, roundKeys.data());
        dumpRoundKeys("AES-256 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        uint8_t *d_plain = nullptr, *d_cipher = nullptr;
        CHECK_CUDA(cudaMalloc(&d_plain, 16));
        CHECK_CUDA(cudaMalloc(&d_cipher, 16));
        CHECK_CUDA(cudaMemcpy(d_plain, plaintext, 16, cudaMemcpyHostToDevice));
        uint64_t ctrHi = 0, ctrLo = 0;
        packCtr(iv, ctrLo, ctrHi);
        aes256_ctr_encrypt<<<1,1>>>(d_plain, d_cipher, 1, ctrLo, ctrHi);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_cipher, 16, cudaMemcpyDeviceToHost));
        bool pass = (std::memcmp(cipher, expectedCipher, 16) == 0);
        printf("[TEST] %s encryption : %s\n", testName, pass ? "PASS" : "FAIL");
        if (!pass) {
            printHex("  Exp", expectedCipher, 16);
            printHex("  Got", cipher, 16);
        }
        allPassed &= pass;
        uint8_t result[16];
        CHECK_CUDA(cudaMemcpy(d_plain, cipher, 16, cudaMemcpyHostToDevice));
        aes256_ctr_decrypt<<<1,1>>>(d_plain, d_cipher, 1, ctrLo, ctrHi);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(result, d_cipher, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(result, plaintext, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", result, 16);
        }
        allPassed &= decPass;
        cudaFree(d_plain);
        cudaFree(d_cipher);
    }
    NVTX_POP();

    // ---------- Test 5: AES-128-GCM single block (NIST SP 800-38D Test Case C.2) ----------
    NVTX_PUSH("KAT_AES128_GCM");
    {
        const char *testName = "AES-128-GCM";
        // Test Case C.2: Key = 128-bit all-zero, IV = 96-bit all-zero, Plaintext = 16 bytes all-zero
        uint8_t key[16] = {0};
        uint8_t iv[12] = {0};
        uint8_t plaintext[16] = {0};
        uint8_t expectedCipher[16] = {
            0x03, 0x88, 0xDA, 0xCE, 0x60, 0xB6, 0xA3, 0x92,
            0xF3, 0x28, 0xC2, 0xB9, 0x71, 0xB2, 0xFE, 0x78
        };
        uint8_t expectedTag[16] = {
            0xAB, 0x6E, 0x47, 0xD4, 0x2C, 0xEC, 0x13, 0xBD,
            0xF5, 0x3A, 0x67, 0xB2, 0x12, 0x57, 0xBD, 0xDF
        };
        std::vector<uint32_t> roundKeys(44);
        expandKey128(key, roundKeys.data());
        dumpRoundKeys("AES-128 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        uint8_t *d_plain = nullptr, *d_cipher = nullptr, *d_iv = nullptr, *d_tag = nullptr;
        CHECK_CUDA(cudaMalloc(&d_plain, 16));
        CHECK_CUDA(cudaMalloc(&d_cipher, 16));
        CHECK_CUDA(cudaMalloc(&d_iv, 12));
        CHECK_CUDA(cudaMalloc(&d_tag, 16));
        CHECK_CUDA(cudaMemcpy(d_plain, plaintext, 16, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_iv, iv, 12, cudaMemcpyHostToDevice));
        // Launch GCM encryption (use 1 block of 256 threads as defined in kernel)
        aes128_gcm_encrypt<<<1,256>>>(d_plain, d_cipher, 1, d_iv, d_tag);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16], tag[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_cipher, 16, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(tag, d_tag, 16, cudaMemcpyDeviceToHost));
        bool passC = (std::memcmp(cipher, expectedCipher, 16) == 0);
        bool passT = (std::memcmp(tag, expectedTag, 16) == 0);
        printf("[TEST] %s encryption : %s (Ciphertext %s, Auth Tag %s)\n", testName,
               (passC && passT ? "PASS" : "FAIL"),
               passC ? "OK" : "BAD", passT ? "OK" : "BAD");
        if (!passC) {
            printHex("  Exp CT", expectedCipher, 16);
            printHex("  Got CT", cipher, 16);
        }
        if (!passT) {
            printHex("  Exp Tag", expectedTag, 16);
            printHex("  Got Tag", tag, 16);
        }
        allPassed &= (passC && passT);
        // Round-trip decryption: decrypt cipher and verify it matches plaintext and tag
        CHECK_CUDA(cudaMemcpy(d_cipher, cipher, 16, cudaMemcpyHostToDevice));
        // Provide expected tag to verify (though GPU kernel just reproduces tag)
        aes128_gcm_decrypt<<<1,256>>>(d_cipher, d_plain, 1, d_iv, d_tag, d_tag);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t decrypted[16], tag2[16];
        CHECK_CUDA(cudaMemcpy(decrypted, d_plain, 16, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(tag2, d_tag, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(decrypted, plaintext, 16) == 0) && (std::memcmp(tag2, expectedTag, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", decrypted, 16);
            printHex("  Tag-Exp", expectedTag, 16);
            printHex("  Tag-Got", tag2, 16);
        }
        allPassed &= decPass;
        cudaFree(d_plain);
        cudaFree(d_cipher);
        cudaFree(d_iv);
        cudaFree(d_tag);
    }
    NVTX_POP();


    // ---------- Test 6: AES-256-GCM single block (NIST SP 800-38D Test Case C.3) ----------
    NVTX_PUSH("KAT_AES256_GCM");
    {
        const char *testName = "AES-256-GCM";
        // Test Case C.3: Key = 2xC.2 key (256-bit), IV = cafebabefacedbaddecaf888, Plaintext = "Single block msg"
        uint8_t key[32] = {
            0xFE, 0xFF, 0xE9, 0x92, 0x86, 0x65, 0x73, 0x1C, 0x6D, 0x6A, 0x8F, 0x94, 0x67, 0x30, 0x83, 0x08,
            0xFE, 0xFF, 0xE9, 0x92, 0x86, 0x65, 0x73, 0x1C, 0x6D, 0x6A, 0x8F, 0x94, 0x67, 0x30, 0x83, 0x08
        };
        uint8_t iv[12] = {
            0xCA, 0xFE, 0xBA, 0xBE, 0xFA, 0xCE, 0xDB, 0xAD, 0xDE, 0xCA, 0xF8, 0x88
        };
        uint8_t plaintext[16] = {
            0x53, 0x69, 0x6E, 0x67, 0x6C, 0x65, 0x20, 0x62, 0x6C, 0x6F, 0x63, 0x6B, 0x20, 0x6D, 0x73, 0x67
        };
        uint8_t expectedCipher[16] = {
            0x14, 0x5D, 0x8F, 0xC5, 0x8B, 0xB8, 0x6E, 0xD3, 0x5B, 0xF1, 0x21, 0xF8, 0x4D, 0xB5, 0xD8, 0x4C
        };
        uint8_t expectedTag[16] = {
            0x4D, 0x5C, 0x2A, 0x27, 0xC6, 0x4A, 0x62, 0xCF, 0x5A, 0xBD, 0x2B, 0xFB, 0x27, 0x02, 0x3D, 0xB8
        };
        std::vector<uint32_t> roundKeys(60);
        expandKey256(key, roundKeys.data());
        dumpRoundKeys("AES-256 RoundKeys", roundKeys.data(), roundKeys.size());
        init_roundKeys(roundKeys.data(), roundKeys.size());
        uint8_t *d_plain = nullptr, *d_cipher = nullptr, *d_iv = nullptr, *d_tag = nullptr;
        CHECK_CUDA(cudaMalloc(&d_plain, 16));
        CHECK_CUDA(cudaMalloc(&d_cipher, 16));
        CHECK_CUDA(cudaMalloc(&d_iv, 12));
        CHECK_CUDA(cudaMalloc(&d_tag, 16));
        CHECK_CUDA(cudaMemcpy(d_plain, plaintext, 16, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_iv, iv, 12, cudaMemcpyHostToDevice));
        aes256_gcm_encrypt<<<1,256>>>(d_plain, d_cipher, 1, d_iv, d_tag);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t cipher[16], tag[16];
        CHECK_CUDA(cudaMemcpy(cipher, d_cipher, 16, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(tag, d_tag, 16, cudaMemcpyDeviceToHost));
        bool passC = (std::memcmp(cipher, expectedCipher, 16) == 0);
        bool passT = (std::memcmp(tag, expectedTag, 16) == 0);
        printf("[TEST] %s encryption : %s (Ciphertext %s, Auth Tag %s)\n", testName,
               (passC && passT ? "PASS" : "FAIL"),
               passC ? "OK" : "BAD", passT ? "OK" : "BAD");
        if (!passC) {
            printHex("  Exp CT", expectedCipher, 16);
            printHex("  Got CT", cipher, 16);
        }
        if (!passT) {
            printHex("  Exp Tag", expectedTag, 16);
            printHex("  Got Tag", tag, 16);
        }
        allPassed &= (passC && passT);
        // Decrypt round-trip
        CHECK_CUDA(cudaMemcpy(d_cipher, cipher, 16, cudaMemcpyHostToDevice));
        aes256_gcm_decrypt<<<1,256>>>(d_cipher, d_plain, 1, d_iv, d_tag, d_tag);
        CHECK_CUDA(cudaDeviceSynchronize());
        uint8_t decrypted[16], tag2[16];
        CHECK_CUDA(cudaMemcpy(decrypted, d_plain, 16, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(tag2, d_tag, 16, cudaMemcpyDeviceToHost));
        bool decPass = (std::memcmp(decrypted, plaintext, 16) == 0) && (std::memcmp(tag2, expectedTag, 16) == 0);
        printf("[TEST] %s decryption round-trip : %s\n", testName, decPass ? "PASS" : "FAIL");
        if (!decPass) {
            printHex("  Dec-Exp", plaintext, 16);
            printHex("  Dec-Got", decrypted, 16);
            printHex("  Tag-Exp", expectedTag, 16);
            printHex("  Tag-Got", tag2, 16);
        }
        allPassed &= decPass;
        cudaFree(d_plain);
        cudaFree(d_cipher);
        cudaFree(d_iv);
        cudaFree(d_tag);
    }
    NVTX_POP();

    printf("All KAT tests %s.\n\n", allPassed ? "PASSED" : "FAILED");

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
            if (keyBits == 128) {
                expandKey128(key.data(), roundKeys.data());
                dumpRoundKeys("AES-128 RoundKeys", roundKeys.data(), roundKeys.size());
            } else {
                expandKey256(key.data(), roundKeys.data());
                dumpRoundKeys("AES-256 RoundKeys", roundKeys.data(), roundKeys.size());
            }
            init_roundKeys(roundKeys.data(), (int) roundKeys.size());
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


    if (d_plain) cudaFree(d_plain);
    if (d_cipher) cudaFree(d_cipher);
    if (d_tag) cudaFree(d_tag);
}

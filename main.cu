/*
// =============================================================
//  Unified CUDA‑AES Test Harness (main.cu)
//  -------------------------------------------------------------
//  * Supports ECB, CTR, and GCM modes with 128‑ and 256‑bit keys
//  * Dispatches to GPU kernels implemented in:
//      - aes128_ecb.cu, aes256_ecb.cu
//      - aes128_ctr.cu, aes256_ctr.cu
//      - aes128_gcm.cu, aes256_gcm.cu
//  * Measures GPU throughput with CUDA events
//  * Optionally validates against OpenSSL (if available)
//  -------------------------------------------------------------
//  Build:  add this file and the mode kernels to CMake.  Ensure
//          OPENSSL_ROOT and libraries are visible if you enable
//          CPU validation.
// =============================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>

// --------------------------------------------------
//  Optional CPU reference (OpenSSL 3.x needed)
// --------------------------------------------------
#ifdef USE_OPENSSL
#include <openssl/evp.h>
#pragma comment(lib, "libcrypto")   // MSVC users: link crypto
#endif

// --------------------------------------------------
//  Forward declarations supplied by each mode file
// --------------------------------------------------
extern void expandKey128(const uint8_t *key16, uint32_t *roundKeys44);
extern void expandKey256(const uint8_t *key32, uint32_t *roundKeys60);

// GPU kernels (launch signatures) – implemented in *.cu files
extern __global__ void aes128_ecb_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t);
extern __global__ void aes256_ecb_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t);
extern __global__ void aes128_ctr_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t, uint64_t, uint64_t);
extern __global__ void aes256_ctr_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t, uint64_t, uint64_t);
extern __global__ void aes128_gcm_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t, const uint8_t* __restrict__, uint8_t* __restrict__);
extern __global__ void aes256_gcm_encrypt(const uint8_t* __restrict__, uint8_t* __restrict__, size_t, const uint8_t* __restrict__, uint8_t* __restrict__);

// --------------------------------------------------
//  Shared device constants (defined once in a .cu)
// --------------------------------------------------
extern void init_T_tables();                        // uploads S‑box + T‑tables
extern void init_roundKeys(const uint32_t *rk, int words);  // copies to const mem
extern void init_gcm_powers(const uint32_t *rk, int nRounds);

// Device constant array must be sized for AES‑256 (max 60 words)
//__device__ __constant__ uint32_t d_roundKeys[60];

// AES S-box (host-side)
static const uint8_t h_sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
  };


// Galois field multiplication by 2
inline uint8_t xtime(uint8_t x) {
    return (x << 1) ^ (x & 0x80 ? 0x1b : 0x00);
}

// Host copies of T-tables
typedef uint32_t u32;
u32 h_T0[256], h_T1[256], h_T2[256], h_T3[256];

// Device constant memory (declared once)
__device__ __constant__ uint8_t d_sbox[256];
__device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];


// --------------------------------------------------
//  Simple helpers
// --------------------------------------------------
static bool hexToBytes(const std::string &hex, std::vector<uint8_t> &out)
{
    if (hex.size() % 2) return false;
    out.resize(hex.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        unsigned int byte;
        if (sscanf(hex.substr(i*2,2).c_str(), "%02x", &byte) != 1) return false;
        out[i] = static_cast<uint8_t>(byte);
    }
    return true;
}

static void checkCuda(cudaError_t err, const char *where)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void init_T_tables() {
    // Copy S-box to device constant memory
    checkCuda(cudaMemcpyToSymbol(d_sbox, h_sbox, sizeof(h_sbox)), "memcpy S-box");

    for (int i = 0; i < 256; ++i) {
        uint8_t s  = h_sbox[i];
        uint8_t s2 = xtime(s);        // 2 · s   (in GF(2^8))
        uint8_t s3 = s2 ^ s;          // 3 · s

        /*  little-endian order :  [   3·s |   s |   s |   2·s ]
        h_T0[i] =  (uint32_t)s2        |
                  ((uint32_t)s  <<  8) |
                  ((uint32_t)s  << 16) |
                  ((uint32_t)s3 << 24);

        h_T1[i] =  (uint32_t)s3        |
                  ((uint32_t)s2 <<  8) |
                  ((uint32_t)s  << 16) |
                  ((uint32_t)s  << 24);

        h_T2[i] =  (uint32_t)s         |
                  ((uint32_t)s3 <<  8) |
                  ((uint32_t)s2 << 16) |
                  ((uint32_t)s  << 24);

        h_T3[i] =  (uint32_t)s         |
                  ((uint32_t)s  <<  8) |
                  ((uint32_t)s3 << 16) |
                  ((uint32_t)s2 << 24);
    }


    // Copy T-tables to device constant memory
    checkCuda(cudaMemcpyToSymbol(d_T0, h_T0, sizeof(h_T0)), "memcpy T0");
    checkCuda(cudaMemcpyToSymbol(d_T1, h_T1, sizeof(h_T1)), "memcpy T1");
    checkCuda(cudaMemcpyToSymbol(d_T2, h_T2, sizeof(h_T2)), "memcpy T2");
    checkCuda(cudaMemcpyToSymbol(d_T3, h_T3, sizeof(h_T3)), "memcpy T3");
}

// right after your init_T_tables(), before main():
void init_roundKeys(const uint32_t *rk, int words)
{
    // copy 'words' 32-bit words into the device constant array d_roundKeys
    cudaMemcpyToSymbol(d_roundKeys, rk, words * sizeof(uint32_t));
}



// At top of main.cu, below your S-box & xtime:
void expandKey128(const uint8_t *key, uint32_t *rk)
{
    static const uint8_t rcon[10] = {
      0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
    };
    // initial 4 words
    for(int i=0;i<4;i++) rk[i] = ((const uint32_t*)key)[i];
    // use host S-box to do SubWord:
    for(int i=4, rc=0;i<44;i++){
      uint32_t tmp = rk[i-1];
      if((i & 3)==0){
        // rotate
        tmp = (tmp<<8)|(tmp>>24);
        // apply S-box on each byte
        uint8_t *b = (uint8_t*)&tmp;
        tmp = (uint32_t)h_sbox[b[0]]<<24
            | (uint32_t)h_sbox[b[1]]<<16
            | (uint32_t)h_sbox[b[2]]<< 8
            | (uint32_t)h_sbox[b[3]];
        // xor Rcon
        tmp ^= (uint32_t)rcon[rc++] << 24;
      }
      rk[i] = rk[i-4] ^ tmp;
    }
}

// ---- 256-bit key schedule (60 words, 14 rounds) ----
void expandKey256(const uint8_t *key, uint32_t *rk)
{
    static const uint8_t rcon[7] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40};
    // copy first 8 words
    for (int i = 0; i < 8; ++i) rk[i] = ((const uint32_t*)key)[i];

    int rc = 0;
    for (int i = 8; i < 60; ++i)
    {
        uint32_t tmp = rk[i-1];
        if (i % 8 == 0)
        {
            // RotWord + SubWord
            tmp = (tmp << 8) | (tmp >> 24);
            uint8_t *b = (uint8_t*)&tmp;
            tmp = (uint32_t)h_sbox[b[0]]<<24 | (uint32_t)h_sbox[b[1]]<<16 |
                  (uint32_t)h_sbox[b[2]]<< 8 | (uint32_t)h_sbox[b[3]];
            tmp ^= (uint32_t)rcon[rc++] << 24;
        }
        else if (i % 8 == 4)
        {
            uint8_t *b = (uint8_t*)&tmp;
            tmp = (uint32_t)h_sbox[b[0]]<<24 | (uint32_t)h_sbox[b[1]]<<16 |
                  (uint32_t)h_sbox[b[2]]<< 8 | (uint32_t)h_sbox[b[3]];
        }
        rk[i] = rk[i-8] ^ tmp;
    }
}


/*
// --------------------------------------------------
//  Entry‑point
//  Usage: cuda_aes <ecb|ctr|gcm> <128|256> <MiB> [iv_hex]
// --------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <ecb|ctr|gcm> <128|256> <MiB> [iv_hex_for_ctr/gcm]" << std::endl;
        return -1;
    }
    std::string modeStr = argv[1];
    int keyBits = std::stoi(argv[2]);
    int messageMiB = std::stoi(argv[3]);

    if ((modeStr != "ecb" && modeStr != "ctr" && modeStr != "gcm") || (keyBits != 128 && keyBits != 256)) {
        std::cerr << "Invalid mode or key size." << std::endl;
        return -1;
    }

    // --------------------------------------------------
    //  Select GPU
    // --------------------------------------------------
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    init_T_tables();

    // --------------------------------------------------
    //  Key + IV handling
    // --------------------------------------------------
    std::vector<uint8_t> key(keyBits / 8);
    std::mt19937_64 rng(std::random_device{}());
    for (auto &b : key) b = uint8_t(rng() & 0xFF);

    std::vector<uint32_t> roundKeys((keyBits==128)?44:60);
    if (keyBits==128)   expandKey128(key.data(), roundKeys.data());
    else                expandKey256(key.data(), roundKeys.data());
    init_roundKeys(roundKeys.data(), (int)roundKeys.size());
    init_gcm_powers(roundKeys.data(), (keyBits==128)?10:14);

    std::vector<uint8_t> iv(16, 0); // default 0 IV
    if (modeStr != "ecb") {
        if (argc >= 5) {
            if (!hexToBytes(argv[4], iv)) {
                std::cerr << "Invalid IV hex string." << std::endl;
                return -1;
            }
            if (iv.size() != 16 && iv.size()!=12) {
                std::cerr << "IV must be 12 or 16 bytes (hex length 24 or 32)." << std::endl;
                return -1;
            }
        } else {
            for (auto &b : iv) b = uint8_t(rng() & 0xFF);
        }
    }

    // --------------------------------------------------
    //  Allocate buffers & populate plaintext
    // --------------------------------------------------
    size_t dataBytes = size_t(messageMiB) << 20;           // MiB → bytes
    size_t numBlocks = (dataBytes + 15) / 16;              // AES block count
    dataBytes = numBlocks * 16;                            // round up

    std::vector<uint8_t> h_plain(dataBytes);
    for (auto &b : h_plain) b = uint8_t(rng() & 0xFF);
    std::vector<uint8_t> h_cipher(dataBytes);

    uint8_t *d_plain=nullptr, *d_cipher=nullptr, *d_tag=nullptr;
    checkCuda(cudaMalloc(&d_plain, dataBytes), "cudaMalloc plain");
    checkCuda(cudaMalloc(&d_cipher, dataBytes), "cudaMalloc cipher");
    checkCuda(cudaMemcpy(d_plain, h_plain.data(), dataBytes, cudaMemcpyHostToDevice), "memcpy H2D");

    if (modeStr == "gcm") {
        checkCuda(cudaMalloc(&d_tag, 16), "cudaMalloc tag");
    }

    // --------------------------------------------------
    //  Launch kernel + timing
    // --------------------------------------------------
    dim3 block(256);
    dim3 grid((numBlocks + block.x - 1) / block.x);
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (modeStr == "ecb" && keyBits==128)
        aes128_ecb_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks);
    else if (modeStr == "ecb" && keyBits==256)
        aes256_ecb_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks);
    else if (modeStr == "ctr" && keyBits==128) {
        uint64_t ctrLo=0, ctrHi=0; memcpy(&ctrLo, iv.data()+8, 8); memcpy(&ctrHi, iv.data(), 8);
        aes128_ctr_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks, ctrLo, ctrHi);
    }
    else if (modeStr == "ctr" && keyBits==256) {
        uint64_t ctrLo=0, ctrHi=0; memcpy(&ctrLo, iv.data()+8, 8); memcpy(&ctrHi, iv.data(), 8);
        aes256_ctr_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks, ctrLo, ctrHi);
    }
    else if (modeStr == "gcm" && keyBits==128)
        aes128_gcm_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks, iv.data(), d_tag);
    else if (modeStr == "gcm" && keyBits==256)
        aes256_gcm_encrypt<<<grid, block>>>(d_plain, d_cipher, numBlocks, iv.data(), d_tag);
    else {
        std::cerr << "Unsupported mode/key combo." << std::endl; return -1;
    }
    cudaEventRecord(stop);
    checkCuda(cudaGetLastError(), "kernel launch");
    cudaEventSynchronize(stop);

    float ms=0; cudaEventElapsedTime(&ms, start, stop);
    double throughput = (double)dataBytes / (1<<30) / (ms/1e3); // GiB/s
    std::cout << "[GPU] " << modeStr << "‑" << keyBits << " processed "
              << (double)dataBytes/(1<<20) << " MiB in " << ms << " ms => "
              << throughput << " GiB/s" << std::endl;

    // --------------------------------------------------
    //  Copy back ciphertext (+ tag) for optional validation
    // --------------------------------------------------
    checkCuda(cudaMemcpy(h_cipher.data(), d_cipher, dataBytes, cudaMemcpyDeviceToHost), "memcpy D2H");

#ifdef USE_OPENSSL
    {
        const EVP_CIPHER *cipher = nullptr;
        if (modeStr=="ecb" && keyBits==128) cipher = EVP_aes_128_ecb();
        else if (modeStr=="ecb" && keyBits==256) cipher = EVP_aes_256_ecb();
        else if (modeStr=="ctr" && keyBits==128) cipher = EVP_aes_128_ctr();
        else if (modeStr=="ctr" && keyBits==256) cipher = EVP_aes_256_ctr();
        else if (modeStr=="gcm" && keyBits==128) cipher = EVP_aes_128_gcm();
        else if (modeStr=="gcm" && keyBits==256) cipher = EVP_aes_256_gcm();
        if (!cipher) {
            std::cerr << "[CPU] Missing cipher in OpenSSL." << std::endl;
        } else {
            std::vector<uint8_t> cpu_out(dataBytes);
            EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
            EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr);
            EVP_CIPHER_CTX_set_padding(ctx, 0);
            EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), (modeStr=="ecb"?nullptr:iv.data()));
            int outLen=0, total=0;
            EVP_EncryptUpdate(ctx, cpu_out.data(), &outLen, h_plain.data(), (int)dataBytes);
            total += outLen;
            EVP_EncryptFinal_ex(ctx, cpu_out.data()+total, &outLen);
            total += outLen;
            EVP_CIPHER_CTX_free(ctx);
            bool ok = (cpu_out == h_cipher);
            std::cout << (ok?"[✓]":"[✗]") << " GPU output matches OpenSSL CPU" << std::endl;
        }
    }
#endif

    // --------------------------------------------------
    //  Cleanup
    // --------------------------------------------------
    cudaFree(d_plain); cudaFree(d_cipher); if(d_tag) cudaFree(d_tag);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
*/
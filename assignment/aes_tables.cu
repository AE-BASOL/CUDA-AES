#include "aes_common.h"
#include <cstdio>
#include <cstdlib>

// Helper to load 32-bit words in big-endian order regardless of host endianness
static inline uint32_t load_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
}

// ----------------------------------------------------------
// AES S-box (256-byte substitution box) and its inverse
// These arrays are placed in constant memory on the GPU for fast access.
// Only this module defines them to avoid multiple-definition errors.
// ----------------------------------------------------------
__device__ __constant__ uint8_t d_sbox[256];
__device__ __constant__ uint8_t d_inv_sbox[256];

// AES T-tables for encryption (combine SubBytes+ShiftRows+MixColumns)
__device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
// AES U-tables for decryption (combine InvSubBytes+InvShiftRows+InvMixColumns)
__device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];

// Device round key array (max 60 32-bit words for AES-256).
// All key schedules (128/256-bit) are stored here; AES-128 uses first 44 words, AES-256 uses 60.
__device__ __constant__ uint32_t d_roundKeys[60];




// Host copies of the S-box and T/U tables (filled in init_T_tables).
static const uint8_t h_sbox[256] = {
    // AES S-box values 0x00-0xFF (256 entries)
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5, 0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0, 0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc, 0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a, 0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0, 0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b, 0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85, 0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5, 0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17, 0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88, 0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c, 0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9, 0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6, 0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e, 0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94, 0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68, 0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};
static uint8_t h_inv_sbox[256];    // will be filled in init_T_tables()
static uint32_t h_T0[256], h_T1[256], h_T2[256], h_T3[256];
static uint32_t h_U0[256], h_U1[256], h_U2[256], h_U3[256];

// Galois field multiplication helper (for generating U-tables)
inline uint8_t xtime(uint8_t x) {
    // Multiply by 2 in GF(2^8)
    return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00));
}

static inline uint8_t mul2(uint8_t x)  { return xtime(x); }
static inline uint8_t mul3(uint8_t x)  { return (uint8_t)(xtime(x) ^ x); }
static inline uint8_t mul4(uint8_t x)  { return xtime(mul2(x)); }
static inline uint8_t mul8(uint8_t x)  { return xtime(mul4(x)); }
static inline uint8_t mul9(uint8_t x)  { return (uint8_t)(mul8(x) ^ x); }
static inline uint8_t mul11(uint8_t x) { return (uint8_t)(mul8(x) ^ mul2(x) ^ x); }
static inline uint8_t mul13(uint8_t x) { return (uint8_t)(mul8(x) ^ mul4(x) ^ x); }
static inline uint8_t mul14(uint8_t x) { return (uint8_t)(mul8(x) ^ mul4(x) ^ mul2(x)); }

// ----------------------------------------------------------
// init_T_tables()
//  - Computes the inverse S-box, encryption T-tables (h_T0..h_T3)
//    and decryption U-tables (h_U0..h_U3) on the host.
//  - Uploads S-box, inverse S-box, and all tables to device constant memory.
// ----------------------------------------------------------
void init_T_tables() {
    // Compute inverse S-box by inverting the h_sbox array
    for (int i = 0; i < 256; ++i) {
        uint8_t s = h_sbox[i];
        h_inv_sbox[s] = (uint8_t)i;
    }

    // Compute encryption T-tables and decryption U-tables
    for (int i = 0; i < 256; ++i) {
        uint8_t s = h_sbox[i];
        uint32_t w = ((uint32_t)mul2(s) << 24) |
                     ((uint32_t)s       << 16) |
                     ((uint32_t)s       << 8)  |
                     (uint32_t)mul3(s);
        h_T0[i] = w;
        h_T1[i] = (w << 8)  | (w >> 24);
        h_T2[i] = (w << 16) | (w >> 16);
        h_T3[i] = (w << 24) | (w >> 8);

        uint8_t v = h_inv_sbox[i];
        uint32_t dw = ((uint32_t)mul14(v) << 24) |
                      ((uint32_t)mul9(v)  << 16) |
                      ((uint32_t)mul13(v) << 8)  |
                      (uint32_t)mul11(v);
        h_U0[i] = dw;
        h_U1[i] = (dw << 8)  | (dw >> 24);
        h_U2[i] = (dw << 16) | (dw >> 16);
        h_U3[i] = (dw << 24) | (dw >> 8);
    }

    // Copy S-box and inverse S-box to device constant memory
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_sbox, h_sbox, sizeof(h_sbox));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(d_sbox) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpyToSymbol(d_inv_sbox, h_inv_sbox, sizeof(h_inv_sbox));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(d_inv_sbox) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Copy T-tables and U-tables to device constant memory
    err = cudaMemcpyToSymbol(d_T0, h_T0, sizeof(h_T0));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_T1, h_T1, sizeof(h_T1));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_T2, h_T2, sizeof(h_T2));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_T3, h_T3, sizeof(h_T3));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_U0, h_U0, sizeof(h_U0));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_U1, h_U1, sizeof(h_U1));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_U2, h_U2, sizeof(h_U2));
    if (err == cudaSuccess) err = cudaMemcpyToSymbol(d_U3, h_U3, sizeof(h_U3));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(T/U tables) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------
// init_roundKeys()
//  - Copies the expanded round key schedule (nWords 32-bit words)
//    from host memory to the device constant memory array d_roundKeys.
// ----------------------------------------------------------
void init_roundKeys(const uint32_t *rk, int nWords) {
    cudaError_t err = cudaMemcpyToSymbol(d_roundKeys, rk, nWords * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol(d_roundKeys) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------------------
// expandKey128() and expandKey256()
//  - Expand a 128-bit or 256-bit AES key on the host into the round key schedule.
//  - They utilize the host S-box (h_sbox) and Rcon for key expansion.
// ----------------------------------------------------------
void expandKey128(const uint8_t *key, uint32_t *rk) {
    static const uint8_t Rcon[10] = {
        0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36
    };
    // Copy initial 4 words from original key (big-endian)
    for (int i = 0; i < 4; ++i) {
        rk[i] = load_be32(key + 4 * i);
    }
    // Expand the remaining 40 words (total 44 words for AES-128: 11 round keys)
    for (int i = 4, rc = 0; i < 44; ++i) {
        uint32_t tmp = rk[i - 1];
        if ((i % 4) == 0) {
            tmp = (tmp << 8) | (tmp >> 24);
            tmp = ((uint32_t)h_sbox[(tmp >> 24) & 0xFF] << 24) |
                  ((uint32_t)h_sbox[(tmp >> 16) & 0xFF] << 16) |
                  ((uint32_t)h_sbox[(tmp >> 8)  & 0xFF] << 8)  |
                  ((uint32_t)h_sbox[tmp & 0xFF]);
            tmp ^= (uint32_t)Rcon[rc++] << 24;
        }
        rk[i] = rk[i - 4] ^ tmp;
    }
}

void expandKey256(const uint8_t *key, uint32_t *rk) {
    static const uint8_t Rcon[7] = {
        0x01,0x02,0x04,0x08,0x10,0x20,0x40
    };
    // Copy initial 8 words from original 256-bit key (big-endian)
    for (int i = 0; i < 8; ++i) {
        rk[i] = load_be32(key + 4 * i);
    }
    // Expand remaining words (total 60 words for AES-256: 15 round keys)
    int rc = 0;
    for (int i = 8; i < 60; ++i) {
        uint32_t tmp = rk[i - 1];
        if ((i % 8) == 0) {
            tmp = (tmp << 8) | (tmp >> 24);
            tmp = ((uint32_t)h_sbox[(tmp >> 24) & 0xFF] << 24) |
                  ((uint32_t)h_sbox[(tmp >> 16) & 0xFF] << 16) |
                  ((uint32_t)h_sbox[(tmp >> 8)  & 0xFF] << 8)  |
                  ((uint32_t)h_sbox[tmp & 0xFF]);
            tmp ^= (uint32_t)Rcon[rc++] << 24;
        } else if ((i % 8) == 4) {
            tmp = ((uint32_t)h_sbox[(tmp >> 24) & 0xFF] << 24) |
                  ((uint32_t)h_sbox[(tmp >> 16) & 0xFF] << 16) |
                  ((uint32_t)h_sbox[(tmp >> 8)  & 0xFF] << 8)  |
                  ((uint32_t)h_sbox[tmp & 0xFF]);
        }
        rk[i] = rk[i - 8] ^ tmp;
    }
}

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
__device__ __constant__ uint64_t d_H_pow_hi[32];
__device__ __constant__ uint64_t d_H_pow_lo[32];




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
        uint8_t s2 = xtime(s);       // 2·s in GF(2^8)
        uint8_t s3 = (uint8_t)(s2 ^ s); // 3·s = 2·s ^ s

        // Little-endian format: word bytes [3*s | s | s | 2*s]
        h_T0[i] =  (uint32_t)s2        | ((uint32_t)s  << 8) |
                   ((uint32_t)s  << 16) | ((uint32_t)s3 << 24);
        h_T1[i] =  (uint32_t)s3        | ((uint32_t)s2 << 8) |
                   ((uint32_t)s  << 16) | ((uint32_t)s  << 24);
        h_T2[i] =  (uint32_t)s         | ((uint32_t)s3 << 8) |
                   ((uint32_t)s2 << 16) | ((uint32_t)s  << 24);
        h_T3[i] =  (uint32_t)s         | ((uint32_t)s  << 8) |
                   ((uint32_t)s3 << 16) | ((uint32_t)s2 << 24);

        // Compute values for inverse S-box
        uint8_t v = h_inv_sbox[i];
        // For decryption MixColumns multipliers: 9, 11, 13, 14
        uint8_t v2  = xtime(v);
        uint8_t v4  = xtime(v2);
        uint8_t v8  = xtime(v4);
        uint8_t v9  = (uint8_t)(v8 ^ v);            // 9·v = v8 ^ v
        uint8_t v11 = (uint8_t)(v8 ^ v2 ^ v);       // 11·v
        uint8_t v13 = (uint8_t)(v8 ^ v4 ^ v);       // 13·v
        uint8_t v14 = (uint8_t)(v8 ^ v4 ^ v2);      // 14·v

        // Little-endian format for decrypt: [14*v | 9*v | 13*v | 11*v]
        h_U0[i] =  (uint32_t)v14       | ((uint32_t)v9  << 8) |
                   ((uint32_t)v13 << 16) | ((uint32_t)v11 << 24);
        h_U1[i] =  (uint32_t)v11       | ((uint32_t)v14 << 8) |
                   ((uint32_t)v9  << 16) | ((uint32_t)v13 << 24);
        h_U2[i] =  (uint32_t)v13       | ((uint32_t)v11 << 8) |
                   ((uint32_t)v14 << 16) | ((uint32_t)v9  << 24);
        h_U3[i] =  (uint32_t)v9        | ((uint32_t)v13 << 8) |
                   ((uint32_t)v11 << 16) | ((uint32_t)v14 << 24);
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
            // Rotate word
            tmp = (tmp << 8) | (tmp >> 24);
            // Apply S-box to each byte of tmp
            uint8_t *bt = (uint8_t*)&tmp;
            bt[0] = h_sbox[bt[0]];
            bt[1] = h_sbox[bt[1]];
            bt[2] = h_sbox[bt[2]];
            bt[3] = h_sbox[bt[3]];
            // XOR with round constant
            bt[0] ^= Rcon[rc++];
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
            // RotWord + SubWord + Rcon
            tmp = (tmp << 8) | (tmp >> 24);
            uint8_t *bt = (uint8_t*)&tmp;
            bt[0] = h_sbox[bt[0]];
            bt[1] = h_sbox[bt[1]];
            bt[2] = h_sbox[bt[2]];
            bt[3] = h_sbox[bt[3]];
            bt[0] ^= Rcon[rc++];
        } else if ((i % 8) == 4) {
            // SubWord (no rotation, for 256-bit key schedule)
            uint8_t *bt = (uint8_t*)&tmp;
            bt[0] = h_sbox[bt[0]];
            bt[1] = h_sbox[bt[1]];
            bt[2] = h_sbox[bt[2]];
            bt[3] = h_sbox[bt[3]];
        }
        rk[i] = rk[i - 8] ^ tmp;
    }
}

// ----------------------------------------------------------
//  Host-side GF(2^128) multiply (same as device gf_mul128)
// ----------------------------------------------------------
static inline void gf_mul128_host(uint64_t &Ah, uint64_t &Al,
                                  uint64_t Bh, uint64_t Bl) {
    uint64_t Zh = 0ull, Zl = 0ull;
    uint64_t Vh = Bh, Vl = Bl;
    const uint64_t R = 0xE100000000000000ULL;
    for (int i = 0; i < 128; ++i) {
        if (Al & 1ULL) { Zh ^= Vh; Zl ^= Vl; }
        bool carry = (Vl & 1ULL);
        Vl = (Vl >> 1) | (Vh << 63);
        Vh >>= 1;
        if (carry) Vh ^= R;
        Al = (Al >> 1) | (Ah << 63);
        Ah >>= 1;
    }
    Ah = Zh; Al = Zl;
}

// ----------------------------------------------------------
//  init_gcm_powers()
//    - Computes H and its powers-of-two using the supplied round keys
//    - Copies results to device constant memory (d_H_pow_hi/lo)
// ----------------------------------------------------------
void init_gcm_powers(const uint32_t *rk, int nRounds) {
    // Compute H = AES_encrypt(0^128)
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
    for (int r = 1; r < nRounds; ++r) {
        t0 = h_T0[s0 & 0xFF] ^ h_T1[(s1>>8)&0xFF] ^ h_T2[(s2>>16)&0xFF] ^ h_T3[(s3>>24)&0xFF] ^ rk[4*r+0];
        t1 = h_T0[s1 & 0xFF] ^ h_T1[(s2>>8)&0xFF] ^ h_T2[(s3>>16)&0xFF] ^ h_T3[(s0>>24)&0xFF] ^ rk[4*r+1];
        t2 = h_T0[s2 & 0xFF] ^ h_T1[(s3>>8)&0xFF] ^ h_T2[(s0>>16)&0xFF] ^ h_T3[(s1>>24)&0xFF] ^ rk[4*r+2];
        t3 = h_T0[s3 & 0xFF] ^ h_T1[(s0>>8)&0xFF] ^ h_T2[(s1>>16)&0xFF] ^ h_T3[(s2>>24)&0xFF] ^ rk[4*r+3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }
    // final round uses S-box only
    uint8_t buf[16];
    buf[0]  = h_sbox[s0 & 0xFF];      buf[4]  = h_sbox[(s1>>8)&0xFF];
    buf[8]  = h_sbox[(s2>>16)&0xFF];  buf[12] = h_sbox[(s3>>24)&0xFF];
    buf[1]  = h_sbox[s1 & 0xFF];      buf[5]  = h_sbox[(s2>>8)&0xFF];
    buf[9]  = h_sbox[(s3>>16)&0xFF];  buf[13] = h_sbox[(s0>>24)&0xFF];
    buf[2]  = h_sbox[s2 & 0xFF];      buf[6]  = h_sbox[(s3>>8)&0xFF];
    buf[10] = h_sbox[(s0>>16)&0xFF];  buf[14] = h_sbox[(s1>>24)&0xFF];
    buf[3]  = h_sbox[s3 & 0xFF];      buf[7]  = h_sbox[(s0>>8)&0xFF];
    buf[11] = h_sbox[(s1>>16)&0xFF];  buf[15] = h_sbox[(s2>>24)&0xFF];
    ((uint32_t*)buf)[0] ^= rk[4*nRounds+0];
    ((uint32_t*)buf)[1] ^= rk[4*nRounds+1];
    ((uint32_t*)buf)[2] ^= rk[4*nRounds+2];
    ((uint32_t*)buf)[3] ^= rk[4*nRounds+3];
    uint64_t H_lo = ((uint64_t*)buf)[0];
    uint64_t H_hi = ((uint64_t*)buf)[1];

    // Compute powers H^(2^i)
    uint64_t pow_hi[32], pow_lo[32];
    pow_hi[0] = H_hi; pow_lo[0] = H_lo;
    for (int i = 1; i < 32; ++i) {
        pow_hi[i] = pow_hi[i-1];
        pow_lo[i] = pow_lo[i-1];
        gf_mul128_host(pow_hi[i], pow_lo[i], pow_hi[i-1], pow_lo[i-1]);
    }
    cudaMemcpyToSymbol(d_H_pow_hi, pow_hi, sizeof(pow_hi));
    cudaMemcpyToSymbol(d_H_pow_lo, pow_lo, sizeof(pow_lo));
}

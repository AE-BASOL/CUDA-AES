#include "aes_common.h"

// Extern constant definitions
extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];

// AES-256 has 14 rounds (Nr=14, key schedule 60 words).
/* helpers */
#define B0(x) (((x) >> 24) & 0xFF)
#define B1(x) (((x) >> 16) & 0xFF)
#define B2(x) (((x) >>  8) & 0xFF)
#define B3(x) ((x) & 0xFF)

#define AES_ENC_ROUND(o0,o1,o2,o3, s0,s1,s2,s3, rk)                \
do {                                                               \
    o0 = sh_T0[B0(s0)] ^ sh_T1[B1(s1)] ^                          \
         sh_T2[B2(s2)] ^ sh_T3[B3(s3)] ^ (rk)[0];                 \
    o1 = sh_T0[B0(s1)] ^ sh_T1[B1(s2)] ^                          \
         sh_T2[B2(s3)] ^ sh_T3[B3(s0)] ^ (rk)[1];                 \
    o2 = sh_T0[B0(s2)] ^ sh_T1[B1(s3)] ^                          \
         sh_T2[B2(s0)] ^ sh_T3[B3(s1)] ^ (rk)[2];                 \
    o3 = sh_T0[B0(s3)] ^ sh_T1[B1(s0)] ^                          \
         sh_T2[B2(s1)] ^ sh_T3[B3(s2)] ^ (rk)[3];                 \
} while (0)

#define AES_DEC_ROUND(o0,o1,o2,o3, s0,s1,s2,s3, rk)                \
do {                                                               \
    o0 = d_U0[B0(s0)] ^ d_U1[B1(s3)] ^                            \
         d_U2[B2(s2)] ^ d_U3[B3(s1)] ^ (rk)[0];                   \
    o1 = d_U0[B0(s1)] ^ d_U1[B1(s0)] ^                            \
         d_U2[B2(s3)] ^ d_U3[B3(s2)] ^ (rk)[1];                   \
    o2 = d_U0[B0(s2)] ^ d_U1[B1(s1)] ^                            \
         d_U2[B2(s0)] ^ d_U3[B3(s3)] ^ (rk)[2];                   \
    o3 = d_U0[B0(s3)] ^ d_U1[B1(s2)] ^                            \
         d_U2[B2(s1)] ^ d_U3[B3(s0)] ^ (rk)[3];                   \
} while (0)

__global__ void aes256_ecb_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    __shared__ uint32_t sh_T0[256], sh_T1[256], sh_T2[256], sh_T3[256];
    __shared__ uint8_t  sh_sbox[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sh_T0[i] = d_T0[i];
        sh_T1[i] = d_T1[i];
        sh_T2[i] = d_T2[i];
        sh_T3[i] = d_T3[i];
        sh_sbox[i] = d_sbox[i];
    }
    __syncthreads();
    const uint32_t *rk = d_roundKeys;  // 60 words for AES-256

    // Load 16 bytes of plaintext
    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = inBlock.x;
    uint32_t s1 = inBlock.y;
    uint32_t s2 = inBlock.z;
    uint32_t s3 = inBlock.w;

    // Initial AddRoundKey (round 0)
    s0 ^= rk[0];  s1 ^= rk[1];
    s2 ^= rk[2];  s3 ^= rk[3];

    // Rounds 1-13
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 1; r <= 13; ++r) {
        AES_ENC_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 4*r);
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Final round (round 14: SubBytes + ShiftRows + AddRoundKey)
    const uint8_t *sb = sh_sbox;
    uint32_t r0 = ((uint32_t)sb[B0(s0)] << 24) |
                  ((uint32_t)sb[B1(s1)] << 16) |
                  ((uint32_t)sb[B2(s2)] <<  8) |
                  ((uint32_t)sb[B3(s3)]);
    uint32_t r1 = ((uint32_t)sb[B0(s1)] << 24) |
                  ((uint32_t)sb[B1(s2)] << 16) |
                  ((uint32_t)sb[B2(s3)] <<  8) |
                  ((uint32_t)sb[B3(s0)]);
    uint32_t r2 = ((uint32_t)sb[B0(s2)] << 24) |
                  ((uint32_t)sb[B1(s3)] << 16) |
                  ((uint32_t)sb[B2(s0)] <<  8) |
                  ((uint32_t)sb[B3(s1)]);
    uint32_t r3 = ((uint32_t)sb[B0(s3)] << 24) |
                  ((uint32_t)sb[B1(s0)] << 16) |
                  ((uint32_t)sb[B2(s1)] <<  8) |
                  ((uint32_t)sb[B3(s2)]);

    r0 ^= rk[56];  // Last round key for AES-256 starts at index 56
    r1 ^= rk[57];
    r2 ^= rk[58];
    r3 ^= rk[59];

    uint4 outBlock = make_uint4(r0, r1, r2, r3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;
}

__global__ void aes256_ecb_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    const uint32_t *rk = d_roundKeys;
    const uint32_t *rk_last = rk + 56;  // last round key for AES-256

    // Load ciphertext and initial AddRoundKey with last round key
    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = inBlock.x ^ rk_last[0];
    uint32_t s1 = inBlock.y ^ rk_last[1];
    uint32_t s2 = inBlock.z ^ rk_last[2];
    uint32_t s3 = inBlock.w ^ rk_last[3];

    // Rounds 13-1
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 13; r >= 1; --r) {
        AES_DEC_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 4*r);
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Final round: InvShiftRows + InvSubBytes, then AddRoundKey
    const uint8_t *isbox = d_inv_sbox;
    uint32_t r0 = ((uint32_t)isbox[B0(s0)] << 24) |
                  ((uint32_t)isbox[B1(s1)] << 16) |
                  ((uint32_t)isbox[B2(s2)] <<  8) |
                  ((uint32_t)isbox[B3(s3)]);
    uint32_t r1 = ((uint32_t)isbox[B0(s1)] << 24) |
                  ((uint32_t)isbox[B1(s2)] << 16) |
                  ((uint32_t)isbox[B2(s3)] <<  8) |
                  ((uint32_t)isbox[B3(s0)]);
    uint32_t r2 = ((uint32_t)isbox[B0(s2)] << 24) |
                  ((uint32_t)isbox[B1(s3)] << 16) |
                  ((uint32_t)isbox[B2(s0)] <<  8) |
                  ((uint32_t)isbox[B3(s1)]);
    uint32_t r3 = ((uint32_t)isbox[B0(s3)] << 24) |
                  ((uint32_t)isbox[B1(s0)] << 16) |
                  ((uint32_t)isbox[B2(s1)] <<  8) |
                  ((uint32_t)isbox[B3(s2)]);

    r0 ^= rk[0];
    r1 ^= rk[1];
    r2 ^= rk[2];
    r3 ^= rk[3];

    uint4 outBlock = make_uint4(r0, r1, r2, r3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;
}

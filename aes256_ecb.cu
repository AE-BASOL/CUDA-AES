#include "aes_common.h"

// Extern constant definitions
extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];

// AES-256 has 14 rounds (Nr=14, key schedule 60 words).
#define AES_ENC_ROUND(o0,o1,o2,o3, s0,s1,s2,s3, rk) {                 \
    o0 = sh_T0[(s0      ) & 0xFF] ^ sh_T1[(s1 >>  8) & 0xFF] ^        \
         sh_T2[(s2 >> 16) & 0xFF] ^ sh_T3[(s3 >> 24) & 0xFF] ^ (rk)[0];\
    o1 = sh_T0[(s1      ) & 0xFF] ^ sh_T1[(s2 >>  8) & 0xFF] ^        \
         sh_T2[(s3 >> 16) & 0xFF] ^ sh_T3[(s0 >> 24) & 0xFF] ^ (rk)[1];\
    o2 = sh_T0[(s2      ) & 0xFF] ^ sh_T1[(s3 >>  8) & 0xFF] ^        \
         sh_T2[(s0 >> 16) & 0xFF] ^ sh_T3[(s1 >> 24) & 0xFF] ^ (rk)[2];\
    o3 = sh_T0[(s3      ) & 0xFF] ^ sh_T1[(s0 >>  8) & 0xFF] ^        \
         sh_T2[(s1 >> 16) & 0xFF] ^ sh_T3[(s2 >> 24) & 0xFF] ^ (rk)[3];\
}

#define AES_DEC_ROUND(o0,o1,o2,o3, s0,s1,s2,s3, rk) {                 \
    o0 = d_U0[(s0      ) & 0xFF] ^ d_U1[(s3 >>  8) & 0xFF] ^          \
         d_U2[(s2 >> 16) & 0xFF] ^ d_U3[(s1 >> 24) & 0xFF] ^ (rk)[0]; \
    o1 = d_U0[(s1      ) & 0xFF] ^ d_U1[(s0 >>  8) & 0xFF] ^          \
         d_U2[(s3 >> 16) & 0xFF] ^ d_U3[(s2 >> 24) & 0xFF] ^ (rk)[1]; \
    o2 = d_U0[(s2      ) & 0xFF] ^ d_U1[(s1 >>  8) & 0xFF] ^          \
         d_U2[(s0 >> 16) & 0xFF] ^ d_U3[(s3 >> 24) & 0xFF] ^ (rk)[2]; \
    o3 = d_U0[(s3      ) & 0xFF] ^ d_U1[(s2 >>  8) & 0xFF] ^          \
         d_U2[(s1 >> 16) & 0xFF] ^ d_U3[(s0 >> 24) & 0xFF] ^ (rk)[3]; \
}

__global__ void aes256_ecb_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    __shared__ uint32_t sh_T0[256], sh_T1[256], sh_T2[256], sh_T3[256];
    __shared__ uint8_t  sh_sbox[256];
    if (threadIdx.x < 256) {
        sh_T0[threadIdx.x] = d_T0[threadIdx.x];
        sh_T1[threadIdx.x] = d_T1[threadIdx.x];
        sh_T2[threadIdx.x] = d_T2[threadIdx.x];
        sh_T3[threadIdx.x] = d_T3[threadIdx.x];
        sh_sbox[threadIdx.x] = d_sbox[threadIdx.x];
    }
    __syncthreads();
    const uint32_t *rk = d_roundKeys;  // 60 words for AES-256

    // Load 16 bytes of plaintext
    const uint8_t *blockPtr = in + idx * 16;
    uint32_t s0 = ((const uint32_t*)blockPtr)[0];
    uint32_t s1 = ((const uint32_t*)blockPtr)[1];
    uint32_t s2 = ((const uint32_t*)blockPtr)[2];
    uint32_t s3 = ((const uint32_t*)blockPtr)[3];

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
    uint8_t *dst = out + idx * 16;
    dst[ 0] = sb[ s0        & 0xFF];  dst[ 4] = sb[(s1 >>  8) & 0xFF];
    dst[ 8] = sb[(s2 >> 16) & 0xFF];  dst[12] = sb[(s3 >> 24) & 0xFF];
    dst[ 1] = sb[ s1        & 0xFF];  dst[ 5] = sb[(s2 >>  8) & 0xFF];
    dst[ 9] = sb[(s3 >> 16) & 0xFF];  dst[13] = sb[(s0 >> 24) & 0xFF];
    dst[ 2] = sb[ s2        & 0xFF];  dst[ 6] = sb[(s3 >>  8) & 0xFF];
    dst[10] = sb[(s0 >> 16) & 0xFF];  dst[14] = sb[(s1 >> 24) & 0xFF];
    dst[ 3] = sb[ s3        & 0xFF];  dst[ 7] = sb[(s0 >>  8) & 0xFF];
    dst[11] = sb[(s1 >> 16) & 0xFF];  dst[15] = sb[(s2 >> 24) & 0xFF];

    uint32_t *out32 = (uint32_t*)dst;
    out32[0] ^= rk[56];  // Last round key for AES-256 starts at index 56
    out32[1] ^= rk[57];
    out32[2] ^= rk[58];
    out32[3] ^= rk[59];
}

__global__ void aes256_ecb_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;
    const uint32_t *rk = d_roundKeys;
    const uint32_t *rk_last = rk + 56;  // last round key for AES-256

    // Load ciphertext and initial AddRoundKey with last round key
    const uint8_t *blockPtr = in + idx * 16;
    uint32_t s0 = ((const uint32_t*)blockPtr)[0] ^ rk_last[0];
    uint32_t s1 = ((const uint32_t*)blockPtr)[1] ^ rk_last[1];
    uint32_t s2 = ((const uint32_t*)blockPtr)[2] ^ rk_last[2];
    uint32_t s3 = ((const uint32_t*)blockPtr)[3] ^ rk_last[3];

    // Rounds 13-1
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 13; r >= 1; --r) {
        AES_DEC_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 4*r);
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Final round: InvShiftRows + InvSubBytes, then AddRoundKey
    const uint8_t *isbox = d_inv_sbox;
    uint8_t *dst = out + idx * 16;
    dst[ 0] = isbox[ s0        & 0xFF];
    dst[ 1] = isbox[ s1        & 0xFF];
    dst[ 2] = isbox[ s2        & 0xFF];
    dst[ 3] = isbox[ s3        & 0xFF];
    dst[ 4] = isbox[(s3 >>  8) & 0xFF];
    dst[ 5] = isbox[(s0 >>  8) & 0xFF];
    dst[ 6] = isbox[(s1 >>  8) & 0xFF];
    dst[ 7] = isbox[(s2 >>  8) & 0xFF];
    dst[ 8] = isbox[(s2 >> 16) & 0xFF];
    dst[ 9] = isbox[(s3 >> 16) & 0xFF];
    dst[10] = isbox[(s0 >> 16) & 0xFF];
    dst[11] = isbox[(s1 >> 16) & 0xFF];
    dst[12] = isbox[(s1 >> 24) & 0xFF];
    dst[13] = isbox[(s2 >> 24) & 0xFF];
    dst[14] = isbox[(s3 >> 24) & 0xFF];
    dst[15] = isbox[(s0 >> 24) & 0xFF];

    uint32_t *out32 = (uint32_t*)dst;
    out32[0] ^= rk[0];
    out32[1] ^= rk[1];
    out32[2] ^= rk[2];
    out32[3] ^= rk[3];
}

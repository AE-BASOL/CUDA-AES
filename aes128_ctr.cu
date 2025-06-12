#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

// AES-CTR encryption and decryption are identical (XOR with keystream).
// Each thread will generate keystream by encrypting a unique counter value and XOR with input.

// The CTR mode counter is 128-bit, provided as (ctrLo, ctrHi) where ctrLo contains the low 64 bits
// (including the 32-bit initial counter in its low half) and ctrHi the upper 64 bits (often derived from IV).
// This kernel processes nBlocks blocks, incrementing the counter for each block.

__global__ void aes128_ctr_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
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
    const uint32_t *rk = d_roundKeys;  // 44 words for AES-128

    // Compute this thread's counter value by adding idx to the 128-bit counter (ctrHi:ctrLo)
    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi;
    if (ctr_lo < ctrLo) { // handle carry
        ctr_hi += 1ULL;
    }

    // Prepare 4-word state from 128-bit counter (assuming little-endian interpretation)
    uint32_t s0 = (uint32_t)(ctr_lo >>  0);
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)(ctr_hi >>  0);
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);

    // AES-128 encrypt the counter (similar to ECB encrypt)
    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 1; r <= 9; ++r) {
        t0 = sh_T0[ s0        & 0xFF] ^ sh_T1[(s1 >>  8) & 0xFF] ^
             sh_T2[(s2 >> 16) & 0xFF] ^ sh_T3[(s3 >> 24) & 0xFF] ^ rk[4*r + 0];
        t1 = sh_T0[ s1        & 0xFF] ^ sh_T1[(s2 >>  8) & 0xFF] ^
             sh_T2[(s3 >> 16) & 0xFF] ^ sh_T3[(s0 >> 24) & 0xFF] ^ rk[4*r + 1];
        t2 = sh_T0[ s2        & 0xFF] ^ sh_T1[(s3 >>  8) & 0xFF] ^
             sh_T2[(s0 >> 16) & 0xFF] ^ sh_T3[(s1 >> 24) & 0xFF] ^ rk[4*r + 2];
        t3 = sh_T0[ s3        & 0xFF] ^ sh_T1[(s0 >>  8) & 0xFF] ^
             sh_T2[(s1 >> 16) & 0xFF] ^ sh_T3[(s2 >> 24) & 0xFF] ^ rk[4*r + 3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }
    // Final round (SubBytes + ShiftRows + AddRoundKey) producing keystream block
    const uint8_t *sb = sh_sbox;
    uint32_t k0 = ((uint32_t)sb[ s0        & 0xFF]) |
                  ((uint32_t)sb[ s1        & 0xFF] << 8) |
                  ((uint32_t)sb[ s2        & 0xFF] << 16) |
                  ((uint32_t)sb[ s3        & 0xFF] << 24);
    uint32_t k1 = ((uint32_t)sb[(s1 >>  8) & 0xFF]) |
                  ((uint32_t)sb[(s2 >>  8) & 0xFF] << 8) |
                  ((uint32_t)sb[(s3 >>  8) & 0xFF] << 16) |
                  ((uint32_t)sb[(s0 >>  8) & 0xFF] << 24);
    uint32_t k2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
                  ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
                  ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
                  ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
    uint32_t k3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
                  ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
                  ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
                  ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);

    k0 ^= rk[40];
    k1 ^= rk[41];
    k2 ^= rk[42];
    k3 ^= rk[43];

    uint4 ptBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint4 outBlock = make_uint4(ptBlock.x ^ k0,
                                ptBlock.y ^ k1,
                                ptBlock.z ^ k2,
                                ptBlock.w ^ k3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;
}

__global__ void aes128_ctr_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    // CTR decryption mirrors encryption; generate the same keystream and XOR
    // with the input ciphertext to recover plaintext.  The previous
    // implementation attempted a dynamic kernel launch from within the device
    // which would have resulted in thousands of kernel launches (one per
    // thread).  Instead, run the encryption logic directly here.

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
    const uint32_t *rk = d_roundKeys;  // 44 words for AES-128

    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi;
    if (ctr_lo < ctrLo) ctr_hi += 1ULL;

    uint32_t s0 = (uint32_t)(ctr_lo);
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)(ctr_hi);
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);

    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
#pragma unroll
    for (int r = 1; r <= 9; ++r) {
        t0 = sh_T0[ s0        & 0xFF] ^ sh_T1[(s1 >>  8) & 0xFF] ^
             sh_T2[(s2 >> 16) & 0xFF] ^ sh_T3[(s3 >> 24) & 0xFF] ^ rk[4*r + 0];
        t1 = sh_T0[ s1        & 0xFF] ^ sh_T1[(s2 >>  8) & 0xFF] ^
             sh_T2[(s3 >> 16) & 0xFF] ^ sh_T3[(s0 >> 24) & 0xFF] ^ rk[4*r + 1];
        t2 = sh_T0[ s2        & 0xFF] ^ sh_T1[(s3 >>  8) & 0xFF] ^
             sh_T2[(s0 >> 16) & 0xFF] ^ sh_T3[(s1 >> 24) & 0xFF] ^ rk[4*r + 2];
        t3 = sh_T0[ s3        & 0xFF] ^ sh_T1[(s0 >>  8) & 0xFF] ^
             sh_T2[(s1 >> 16) & 0xFF] ^ sh_T3[(s2 >> 24) & 0xFF] ^ rk[4*r + 3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    const uint8_t *sb = sh_sbox;
    uint32_t k0 = ((uint32_t)sb[ s0        & 0xFF]) |
                  ((uint32_t)sb[ s1        & 0xFF] << 8) |
                  ((uint32_t)sb[ s2        & 0xFF] << 16) |
                  ((uint32_t)sb[ s3        & 0xFF] << 24);
    uint32_t k1 = ((uint32_t)sb[(s1 >>  8) & 0xFF]) |
                  ((uint32_t)sb[(s2 >>  8) & 0xFF] << 8) |
                  ((uint32_t)sb[(s3 >>  8) & 0xFF] << 16) |
                  ((uint32_t)sb[(s0 >>  8) & 0xFF] << 24);
    uint32_t k2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
                  ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
                  ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
                  ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
    uint32_t k3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
                  ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
                  ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
                  ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);

    k0 ^= rk[40];
    k1 ^= rk[41];
    k2 ^= rk[42];
    k3 ^= rk[43];

    uint4 ctBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint4 ptBlock = make_uint4(ctBlock.x ^ k0,
                               ctBlock.y ^ k1,
                               ctBlock.z ^ k2,
                               ctBlock.w ^ k3);
    reinterpret_cast<uint4*>(out)[idx] = ptBlock;
}

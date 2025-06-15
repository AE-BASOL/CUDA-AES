#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

#define CTR_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                                   \
    do {                                                                       \
        o0 = d_T0[(s0) & 0xFF] ^ d_T1[((s1) >> 8) & 0xFF] ^                   \
             d_T2[((s2) >>16) & 0xFF] ^ d_T3[((s3) >>24) & 0xFF] ^ (rk)[0];   \
        o1 = d_T0[(s1) & 0xFF] ^ d_T1[((s2) >> 8) & 0xFF] ^                   \
             d_T2[((s3) >>16) & 0xFF] ^ d_T3[((s0) >>24) & 0xFF] ^ (rk)[1];   \
        o2 = d_T0[(s2) & 0xFF] ^ d_T1[((s3) >> 8) & 0xFF] ^                   \
             d_T2[((s0) >>16) & 0xFF] ^ d_T3[((s1) >>24) & 0xFF] ^ (rk)[2];   \
        o3 = d_T0[(s3) & 0xFF] ^ d_T1[((s0) >> 8) & 0xFF] ^                   \
             d_T2[((s1) >>16) & 0xFF] ^ d_T3[((s2) >>24) & 0xFF] ^ (rk)[3];   \
    } while (0)

// AES-CTR encryption and decryption are identical (XOR with keystream).
// Each thread will generate keystream by encrypting a unique counter value and XOR with input.

// The CTR mode counter is 128-bit, provided as (ctrLo, ctrHi) where ctrLo contains the low 64 bits
// (including the 32-bit initial counter in its low half) and ctrHi the upper 64 bits (often derived from IV).
// This kernel processes nBlocks blocks, incrementing the counter for each block.

__global__ void aes128_ctr_encrypt(const uint8_t *in, uint8_t *out,
                                   size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    if (idx >= nBlocks) return;
    // Use tables directly from constant memory to avoid per-block copies
    const uint32_t *rk = d_roundKeys;  // 44 words for AES-128

    // -------- first counter --------
    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi;
    if (ctr_lo < ctrLo) ctr_hi += 1ULL;

    uint32_t s0 = (uint32_t)ctr_lo;
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)ctr_hi;
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);

    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 8);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +12);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +16);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +20);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +24);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +28);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +32);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +36);
    s0=t0; s1=t1; s2=t2; s3=t3;

    const uint8_t *sb = d_sbox;
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

    k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];

    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint4 outBlock = make_uint4(inBlock.x ^ k0,
                                inBlock.y ^ k1,
                                inBlock.z ^ k2,
                                inBlock.w ^ k3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;

    // -------- second counter --------
    size_t idx2 = idx + stride;
    if (idx2 < nBlocks) {
        ctr_lo = ctrLo + idx2;
        ctr_hi = ctrHi;
        if (ctr_lo < ctrLo) ctr_hi += 1ULL;

        s0 = (uint32_t)ctr_lo;
        s1 = (uint32_t)(ctr_lo >> 32);
        s2 = (uint32_t)ctr_hi;
        s3 = (uint32_t)(ctr_hi >> 32);

        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 8);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +12);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +16);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +20);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +24);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +28);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +32);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +36);
        s0=t0; s1=t1; s2=t2; s3=t3;

        k0 = ((uint32_t)sb[ s0        & 0xFF]) |
             ((uint32_t)sb[ s1        & 0xFF] << 8) |
             ((uint32_t)sb[ s2        & 0xFF] << 16) |
             ((uint32_t)sb[ s3        & 0xFF] << 24);
        k1 = ((uint32_t)sb[(s1 >>  8) & 0xFF]) |
             ((uint32_t)sb[(s2 >>  8) & 0xFF] << 8) |
             ((uint32_t)sb[(s3 >>  8) & 0xFF] << 16) |
             ((uint32_t)sb[(s0 >>  8) & 0xFF] << 24);
        k2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
             ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
             ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
             ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
        k3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
             ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
             ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
             ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);

        k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];

        inBlock = reinterpret_cast<const uint4*>(in)[idx2];
        outBlock = make_uint4(inBlock.x ^ k0,
                              inBlock.y ^ k1,
                              inBlock.z ^ k2,
                              inBlock.w ^ k3);
        reinterpret_cast<uint4*>(out)[idx2] = outBlock;
    }
}

__global__ void aes128_ctr_decrypt(const uint8_t *in, uint8_t *out,
                                   size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    // CTR decryption mirrors encryption; generate the same keystream and XOR
    // with the input ciphertext to recover plaintext.

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    if (idx >= nBlocks) return;
    // Tables accessed directly from constant memory
    const uint32_t *rk = d_roundKeys;  // 44 words for AES-128

    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi;
    if (ctr_lo < ctrLo) ctr_hi += 1ULL;

    uint32_t s0 = (uint32_t)ctr_lo;
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)ctr_hi;
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);

    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0, t1, t2, t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 8);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +12);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +16);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +20);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +24);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +28);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +32);
    s0=t0; s1=t1; s2=t2; s3=t3;
    CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +36);
    s0=t0; s1=t1; s2=t2; s3=t3;

    const uint8_t *sb = d_sbox;
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

    k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];

    uint4 ctBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint4 ptBlock = make_uint4(ctBlock.x ^ k0,
                               ctBlock.y ^ k1,
                               ctBlock.z ^ k2,
                               ctBlock.w ^ k3);
    reinterpret_cast<uint4*>(out)[idx] = ptBlock;

    size_t idx2 = idx + stride;
    if (idx2 < nBlocks) {
        ctr_lo = ctrLo + idx2;
        ctr_hi = ctrHi;
        if (ctr_lo < ctrLo) ctr_hi += 1ULL;

        s0 = (uint32_t)ctr_lo;
        s1 = (uint32_t)(ctr_lo >> 32);
        s2 = (uint32_t)ctr_hi;
        s3 = (uint32_t)(ctr_hi >> 32);

        s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 8);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +12);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +16);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +20);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +24);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +28);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +32);
        s0=t0; s1=t1; s2=t2; s3=t3;
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk +36);
        s0=t0; s1=t1; s2=t2; s3=t3;

        k0 = ((uint32_t)sb[ s0        & 0xFF]) |
             ((uint32_t)sb[ s1        & 0xFF] << 8) |
             ((uint32_t)sb[ s2        & 0xFF] << 16) |
             ((uint32_t)sb[ s3        & 0xFF] << 24);
        k1 = ((uint32_t)sb[(s1 >>  8) & 0xFF]) |
             ((uint32_t)sb[(s2 >>  8) & 0xFF] << 8) |
             ((uint32_t)sb[(s3 >>  8) & 0xFF] << 16) |
             ((uint32_t)sb[(s0 >>  8) & 0xFF] << 24);
        k2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
             ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
             ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
             ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
        k3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
             ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
             ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
             ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);

        k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];

        ctBlock = reinterpret_cast<const uint4*>(in)[idx2];
        ptBlock = make_uint4(ctBlock.x ^ k0,
                             ctBlock.y ^ k1,
                             ctBlock.z ^ k2,
                             ctBlock.w ^ k3);
reinterpret_cast<uint4*>(out)[idx2] = ptBlock;
    }
}

#undef CTR_ROUND

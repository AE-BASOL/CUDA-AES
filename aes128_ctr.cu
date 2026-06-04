#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

#define CTR_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                                   \
    do {                                                                       \
        o0 = sh_T0[(s0) & 0xFF] ^ sh_T1[((s1) >> 8) & 0xFF] ^                   \
             sh_T2[((s2) >>16) & 0xFF] ^ sh_T3[((s3) >>24) & 0xFF] ^ (rk)[0];   \
        o1 = sh_T0[(s1) & 0xFF] ^ sh_T1[((s2) >> 8) & 0xFF] ^                   \
             sh_T2[((s3) >>16) & 0xFF] ^ sh_T3[((s0) >>24) & 0xFF] ^ (rk)[1];   \
        o2 = sh_T0[(s2) & 0xFF] ^ sh_T1[((s3) >> 8) & 0xFF] ^                   \
             sh_T2[((s0) >>16) & 0xFF] ^ sh_T3[((s1) >>24) & 0xFF] ^ (rk)[2];   \
        o3 = sh_T0[(s3) & 0xFF] ^ sh_T1[((s0) >> 8) & 0xFF] ^                   \
             sh_T2[((s1) >>16) & 0xFF] ^ sh_T3[((s2) >>24) & 0xFF] ^ (rk)[3];   \
    } while (0)

// AES-CTR encryption and decryption are identical (XOR with keystream).
// Each thread will generate keystream by encrypting a unique counter value and XOR with input.

// The CTR mode counter is 128-bit, provided as (ctrLo, ctrHi) where ctrLo contains the low 64 bits
// (including the 32-bit initial counter in its low half) and ctrHi the upper 64 bits (often derived from IV).
// This kernel processes nBlocks blocks, incrementing the counter for each block.

__device__ __forceinline__ uint4 aes128_ctr_generate_keystream(uint64_t ctr_lo,
                                                               uint64_t ctr_hi,
                                                               const uint32_t *rk,
                                                               const uint32_t sh_T0[256],
                                                               const uint32_t sh_T1[256],
                                                               const uint32_t sh_T2[256],
                                                               const uint32_t sh_T3[256],
                                                               const uint8_t sh_sbox[256]) {
    uint32_t s0 = static_cast<uint32_t>(ctr_lo);
    uint32_t s1 = static_cast<uint32_t>(ctr_lo >> 32);
    uint32_t s2 = static_cast<uint32_t>(ctr_hi);
    uint32_t s3 = static_cast<uint32_t>(ctr_hi >> 32);

    s0 ^= rk[0];
    s1 ^= rk[1];
    s2 ^= rk[2];
    s3 ^= rk[3];

    uint32_t t0, t1, t2, t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 4);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 8);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 12);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 16);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 20);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 24);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 28);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 32);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    CTR_ROUND(t0, t1, t2, t3, s0, s1, s2, s3, rk + 36);
    s0 = t0; s1 = t1; s2 = t2; s3 = t3;

    const uint8_t *sb = sh_sbox;
    uint32_t k0 = static_cast<uint32_t>(sb[s0 & 0xFF]) |
                  (static_cast<uint32_t>(sb[s1 & 0xFF]) << 8) |
                  (static_cast<uint32_t>(sb[s2 & 0xFF]) << 16) |
                  (static_cast<uint32_t>(sb[s3 & 0xFF]) << 24);
    uint32_t k1 = static_cast<uint32_t>(sb[(s1 >> 8) & 0xFF]) |
                  (static_cast<uint32_t>(sb[(s2 >> 8) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(sb[(s3 >> 8) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(sb[(s0 >> 8) & 0xFF]) << 24);
    uint32_t k2 = static_cast<uint32_t>(sb[(s2 >> 16) & 0xFF]) |
                  (static_cast<uint32_t>(sb[(s3 >> 16) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(sb[(s0 >> 16) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(sb[(s1 >> 16) & 0xFF]) << 24);
    uint32_t k3 = static_cast<uint32_t>(sb[(s3 >> 24) & 0xFF]) |
                  (static_cast<uint32_t>(sb[(s0 >> 24) & 0xFF]) << 8) |
                  (static_cast<uint32_t>(sb[(s1 >> 24) & 0xFF]) << 16) |
                  (static_cast<uint32_t>(sb[(s2 >> 24) & 0xFF]) << 24);

    k0 ^= rk[40];
    k1 ^= rk[41];
    k2 ^= rk[42];
    k3 ^= rk[43];

    return make_uint4(k0, k1, k2, k3);
}

static __device__ __forceinline__ uint32_t bswap32_dev(uint32_t x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}

static __device__ __forceinline__ uint64_t ctr_hi_inc32(uint64_t ctr_hi, size_t blockIndex) {
    const uint32_t nonce_hi = static_cast<uint32_t>(ctr_hi);
    const uint32_t counter_word = static_cast<uint32_t>(ctr_hi >> 32);
    const uint32_t counter_be = bswap32_dev(counter_word);
    const uint32_t next_counter_word = bswap32_dev(counter_be + static_cast<uint32_t>(blockIndex));
    return static_cast<uint64_t>(nonce_hi) | (static_cast<uint64_t>(next_counter_word) << 32);
}

__global__ void aes128_ctr_encrypt(const uint8_t *in, uint8_t *out,
                                   size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    const size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

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
    const uint4 *in4 = reinterpret_cast<const uint4*>(in);
    uint4 *out4 = reinterpret_cast<uint4*>(out);

    for (size_t blockIndex = threadId; blockIndex < nBlocks; blockIndex += stride) {
        const uint64_t ctr_lo = ctrLo;
        const uint64_t ctr_hi = ctr_hi_inc32(ctrHi, blockIndex);

        uint4 keystream = aes128_ctr_generate_keystream(ctr_lo, ctr_hi, rk,
                                                        sh_T0, sh_T1, sh_T2, sh_T3, sh_sbox);

        uint4 inBlock = in4[blockIndex];
        uint4 outBlock = make_uint4(inBlock.x ^ keystream.x,
                                    inBlock.y ^ keystream.y,
                                    inBlock.z ^ keystream.z,
                                    inBlock.w ^ keystream.w);
        out4[blockIndex] = outBlock;
    }
}

__global__ void aes128_ctr_decrypt(const uint8_t *in, uint8_t *out,
                                   size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    // CTR decryption mirrors encryption; generate the same keystream and XOR
    // with the input ciphertext to recover plaintext.

    const size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

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
    const uint4 *in4 = reinterpret_cast<const uint4*>(in);
    uint4 *out4 = reinterpret_cast<uint4*>(out);

    for (size_t blockIndex = threadId; blockIndex < nBlocks; blockIndex += stride) {
        const uint64_t ctr_lo = ctrLo;
        const uint64_t ctr_hi = ctr_hi_inc32(ctrHi, blockIndex);

        uint4 keystream = aes128_ctr_generate_keystream(ctr_lo, ctr_hi, rk,
                                                        sh_T0, sh_T1, sh_T2, sh_T3, sh_sbox);

        uint4 ctBlock = in4[blockIndex];
        uint4 ptBlock = make_uint4(ctBlock.x ^ keystream.x,
                                   ctBlock.y ^ keystream.y,
                                   ctBlock.z ^ keystream.z,
                                   ctBlock.w ^ keystream.w);
        out4[blockIndex] = ptBlock;
    }
}

#undef CTR_ROUND

// AES-128 ECB kernels using the same T-table layout as the CTR
// implementation. This mirrors the approach in Cihangir Tezcan's
// optimized AES kernels and gives identical results while avoiding the
// register heavy MixColumns helpers used previously.
// Each thread still processes two consecutive blocks in a grid stride
// loop so the public kernel signature remains unchanged.

#include <cuda_runtime.h>
#include <stdint.h>
#include "aes_common.h"

// Device constant memory declarations come from aes_common.h

// Macros implementing one AES round using T-tables loaded in shared memory.
// Layout matches the AES-CTR implementation and therefore Cihangir's kernels.
#define ENC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                          \
    do {                                                               \
        o0 = sh_T0[(s0)&0xFF] ^ sh_T1[((s1)>>8)&0xFF] ^               \
             sh_T2[((s2)>>16)&0xFF] ^ sh_T3[((s3)>>24)&0xFF] ^ (rk)[0];\
        o1 = sh_T0[(s1)&0xFF] ^ sh_T1[((s2)>>8)&0xFF] ^               \
             sh_T2[((s3)>>16)&0xFF] ^ sh_T3[((s0)>>24)&0xFF] ^ (rk)[1];\
        o2 = sh_T0[(s2)&0xFF] ^ sh_T1[((s3)>>8)&0xFF] ^               \
             sh_T2[((s0)>>16)&0xFF] ^ sh_T3[((s1)>>24)&0xFF] ^ (rk)[2];\
        o3 = sh_T0[(s3)&0xFF] ^ sh_T1[((s0)>>8)&0xFF] ^               \
             sh_T2[((s1)>>16)&0xFF] ^ sh_T3[((s2)>>24)&0xFF] ^ (rk)[3];\
    } while (0)

#define DEC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                          \
    do {                                                               \
        o0 = sh_U0[(s0)&0xFF] ^ sh_U1[((s3)>>8)&0xFF] ^               \
             sh_U2[((s2)>>16)&0xFF] ^ sh_U3[((s1)>>24)&0xFF] ^ (rk)[0];\
        o1 = sh_U0[(s1)&0xFF] ^ sh_U1[((s0)>>8)&0xFF] ^               \
             sh_U2[((s3)>>16)&0xFF] ^ sh_U3[((s2)>>24)&0xFF] ^ (rk)[1];\
        o2 = sh_U0[(s2)&0xFF] ^ sh_U1[((s1)>>8)&0xFF] ^               \
             sh_U2[((s0)>>16)&0xFF] ^ sh_U3[((s3)>>24)&0xFF] ^ (rk)[2];\
        o3 = sh_U0[(s3)&0xFF] ^ sh_U1[((s2)>>8)&0xFF] ^               \
             sh_U2[((s1)>>16)&0xFF] ^ sh_U3[((s0)>>24)&0xFF] ^ (rk)[3];\
    } while (0)

// ─────────────────────────── Kernels ────────────────────────────────────────
__global__ void aes128_ecb_encrypt(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint4* in4  = reinterpret_cast<const uint4*>(in);
    uint4* out4       = reinterpret_cast<uint4*>(out);
    const uint32_t* rk = d_roundKeys;

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

    for (size_t blk = tid * 2; blk < nBlocks; blk += stride * 2) {
        size_t blk2 = blk + 1;
        uint4 inBlock = in4[blk];
        uint32_t s0 = inBlock.x ^ rk[0];
        uint32_t s1 = inBlock.y ^ rk[1];
        uint32_t s2 = inBlock.z ^ rk[2];
        uint32_t s3 = inBlock.w ^ rk[3];

        uint32_t t0,t1,t2,t3;
#pragma unroll
        for (int r = 4; r <= 36; r += 4) {
            ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + r);
            s0=t0; s1=t1; s2=t2; s3=t3;
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

        k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];
        out4[blk] = make_uint4(k0,k1,k2,k3);

        if (blk2 < nBlocks) {
            inBlock = in4[blk2];
            s0 = inBlock.x ^ rk[0];
            s1 = inBlock.y ^ rk[1];
            s2 = inBlock.z ^ rk[2];
            s3 = inBlock.w ^ rk[3];
            for (int r = 4; r <= 36; r += 4) {
                ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + r);
                s0=t0; s1=t1; s2=t2; s3=t3;
            }
            k0 = ((uint32_t)sb[ s0 & 0xFF]) |
                 ((uint32_t)sb[ s1 & 0xFF] << 8) |
                 ((uint32_t)sb[ s2 & 0xFF] << 16) |
                 ((uint32_t)sb[ s3 & 0xFF] << 24);
            k1 = ((uint32_t)sb[(s1>>8)&0xFF]) |
                 ((uint32_t)sb[(s2>>8)&0xFF] << 8) |
                 ((uint32_t)sb[(s3>>8)&0xFF] << 16) |
                 ((uint32_t)sb[(s0>>8)&0xFF] << 24);
            k2 = ((uint32_t)sb[(s2>>16)&0xFF]) |
                 ((uint32_t)sb[(s3>>16)&0xFF] << 8) |
                 ((uint32_t)sb[(s0>>16)&0xFF] << 16) |
                 ((uint32_t)sb[(s1>>16)&0xFF] << 24);
            k3 = ((uint32_t)sb[(s3>>24)&0xFF]) |
                 ((uint32_t)sb[(s0>>24)&0xFF] << 8) |
                 ((uint32_t)sb[(s1>>24)&0xFF] << 16) |
                 ((uint32_t)sb[(s2>>24)&0xFF] << 24);
            k0 ^= rk[40]; k1 ^= rk[41]; k2 ^= rk[42]; k3 ^= rk[43];
            out4[blk2] = make_uint4(k0,k1,k2,k3);
        }
    }
}

__global__ void aes128_ecb_decrypt(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint4* in4  = reinterpret_cast<const uint4*>(in);
    uint4* out4       = reinterpret_cast<uint4*>(out);
    const uint32_t* rk = d_roundKeys;

    __shared__ uint32_t sh_U0[256], sh_U1[256], sh_U2[256], sh_U3[256];
    __shared__ uint8_t  sh_isbox[256];
    if (threadIdx.x < 256) {
        sh_U0[threadIdx.x] = d_U0[threadIdx.x];
        sh_U1[threadIdx.x] = d_U1[threadIdx.x];
        sh_U2[threadIdx.x] = d_U2[threadIdx.x];
        sh_U3[threadIdx.x] = d_U3[threadIdx.x];
        sh_isbox[threadIdx.x] = d_inv_sbox[threadIdx.x];
    }
    __syncthreads();

    for (size_t blk = tid * 2; blk < nBlocks; blk += stride * 2) {
        size_t blk2 = blk + 1;
        uint4 inBlock = in4[blk];
        uint32_t s0 = inBlock.x ^ rk[40];
        uint32_t s1 = inBlock.y ^ rk[41];
        uint32_t s2 = inBlock.z ^ rk[42];
        uint32_t s3 = inBlock.w ^ rk[43];

        uint32_t t0,t1,t2,t3;
#pragma unroll
        for (int r = 36; r >= 4; r -= 4) {
            DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + r);
            s0=t0; s1=t1; s2=t2; s3=t3;
        }

        const uint8_t *isb = sh_isbox;
        uint32_t k0 = ((uint32_t)isb[ s0        & 0xFF]) |
                      ((uint32_t)isb[ s3        & 0xFF] << 8) |
                      ((uint32_t)isb[ s2        & 0xFF] << 16) |
                      ((uint32_t)isb[ s1        & 0xFF] << 24);
        uint32_t k1 = ((uint32_t)isb[(s1 >>  8) & 0xFF]) |
                      ((uint32_t)isb[(s0 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s3 >>  8) & 0xFF] << 16) |
                      ((uint32_t)isb[(s2 >>  8) & 0xFF] << 24);
        uint32_t k2 = ((uint32_t)isb[(s2 >> 16) & 0xFF]) |
                      ((uint32_t)isb[(s1 >> 16) & 0xFF] << 8) |
                      ((uint32_t)isb[(s0 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s3 >> 16) & 0xFF] << 24);
        uint32_t k3 = ((uint32_t)isb[(s3 >> 24) & 0xFF]) |
                      ((uint32_t)isb[(s2 >> 24) & 0xFF] << 8) |
                      ((uint32_t)isb[(s1 >> 24) & 0xFF] << 16) |
                      ((uint32_t)isb[(s0 >> 24) & 0xFF] << 24);

        k0 ^= rk[0]; k1 ^= rk[1]; k2 ^= rk[2]; k3 ^= rk[3];
        out4[blk] = make_uint4(k0,k1,k2,k3);

        if (blk2 < nBlocks) {
            inBlock = in4[blk2];
            s0 = inBlock.x ^ rk[40];
            s1 = inBlock.y ^ rk[41];
            s2 = inBlock.z ^ rk[42];
            s3 = inBlock.w ^ rk[43];
            for (int r = 36; r >= 4; r -= 4) {
                DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + r);
                s0=t0; s1=t1; s2=t2; s3=t3;
            }
            k0 = ((uint32_t)isb[ s0 & 0xFF]) |
                 ((uint32_t)isb[ s3 & 0xFF] << 8) |
                 ((uint32_t)isb[ s2 & 0xFF] << 16) |
                 ((uint32_t)isb[ s1 & 0xFF] << 24);
            k1 = ((uint32_t)isb[(s1>>8)&0xFF]) |
                 ((uint32_t)isb[(s0>>8)&0xFF] << 8) |
                 ((uint32_t)isb[(s3>>8)&0xFF] << 16) |
                 ((uint32_t)isb[(s2>>8)&0xFF] << 24);
            k2 = ((uint32_t)isb[(s2>>16)&0xFF]) |
                 ((uint32_t)isb[(s1>>16)&0xFF] << 8) |
                 ((uint32_t)isb[(s0>>16)&0xFF] << 16) |
                 ((uint32_t)isb[(s3>>16)&0xFF] << 24);
            k3 = ((uint32_t)isb[(s3>>24)&0xFF]) |
                 ((uint32_t)isb[(s2>>24)&0xFF] << 8) |
                 ((uint32_t)isb[(s1>>24)&0xFF] << 16) |
                 ((uint32_t)isb[(s0>>24)&0xFF] << 24);
            k0 ^= rk[0]; k1 ^= rk[1]; k2 ^= rk[2]; k3 ^= rk[3];
            out4[blk2] = make_uint4(k0,k1,k2,k3);
        }
    }
}

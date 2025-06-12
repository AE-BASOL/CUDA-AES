#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];

#define AES_ENC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk) do { \
    o0 = sh_T0[(s0) & 0xFF] ^ sh_T1[((s1)>>8) & 0xFF] ^ \
         sh_T2[((s2)>>16)&0xFF] ^ sh_T3[((s3)>>24)&0xFF] ^ (rk)[0]; \
    o1 = sh_T0[(s1) & 0xFF] ^ sh_T1[((s2)>>8) & 0xFF] ^ \
         sh_T2[((s3)>>16)&0xFF] ^ sh_T3[((s0)>>24)&0xFF] ^ (rk)[1]; \
    o2 = sh_T0[(s2) & 0xFF] ^ sh_T1[((s3)>>8) & 0xFF] ^ \
         sh_T2[((s0)>>16)&0xFF] ^ sh_T3[((s1)>>24)&0xFF] ^ (rk)[2]; \
    o3 = sh_T0[(s3) & 0xFF] ^ sh_T1[((s0)>>8) & 0xFF] ^ \
         sh_T2[((s1)>>16)&0xFF] ^ sh_T3[((s2)>>24)&0xFF] ^ (rk)[3]; \
} while(0)

#define AES_DEC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk) do { \
    o0 = d_U0[(s0) & 0xFF] ^ d_U1[((s3)>>8)&0xFF] ^ \
         d_U2[((s2)>>16)&0xFF] ^ d_U3[((s1)>>24)&0xFF] ^ (rk)[0]; \
    o1 = d_U0[(s1) & 0xFF] ^ d_U1[((s0)>>8)&0xFF] ^ \
         d_U2[((s3)>>16)&0xFF] ^ d_U3[((s2)>>24)&0xFF] ^ (rk)[1]; \
    o2 = d_U0[(s2) & 0xFF] ^ d_U1[((s1)>>8)&0xFF] ^ \
         d_U2[((s0)>>16)&0xFF] ^ d_U3[((s3)>>24)&0xFF] ^ (rk)[2]; \
    o3 = d_U0[(s3) & 0xFF] ^ d_U1[((s2)>>8)&0xFF] ^ \
         d_U2[((s1)>>16)&0xFF] ^ d_U3[((s0)>>24)&0xFF] ^ (rk)[3]; \
} while(0)

template<int ROUNDS>
__device__ void aes_ecb_encrypt_impl(const uint8_t* __restrict__ in,
                                     uint8_t* __restrict__ out,
                                     size_t nBlocks) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    const uint32_t *rk = d_roundKeys;
    uint4 block = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = block.x;
    uint32_t s1 = block.y;
    uint32_t s2 = block.z;
    uint32_t s3 = block.w;

    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0,t1,t2,t3;
#pragma unroll
    for (int r = 1; r < ROUNDS; ++r) {
        AES_ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4*r);
        s0=t0; s1=t1; s2=t2; s3=t3;
    }
    const uint8_t *sb = sh_sbox;
    uint32_t r0 = ((uint32_t)sb[ s0        & 0xFF]) |
                  ((uint32_t)sb[ s1        & 0xFF] << 8) |
                  ((uint32_t)sb[ s2        & 0xFF] << 16) |
                  ((uint32_t)sb[ s3        & 0xFF] << 24);
    uint32_t r1 = ((uint32_t)sb[(s1 >>  8) & 0xFF]) |
                  ((uint32_t)sb[(s2 >>  8) & 0xFF] << 8) |
                  ((uint32_t)sb[(s3 >>  8) & 0xFF] << 16) |
                  ((uint32_t)sb[(s0 >>  8) & 0xFF] << 24);
    uint32_t r2 = ((uint32_t)sb[(s2 >> 16) & 0xFF]) |
                  ((uint32_t)sb[(s3 >> 16) & 0xFF] << 8) |
                  ((uint32_t)sb[(s0 >> 16) & 0xFF] << 16) |
                  ((uint32_t)sb[(s1 >> 16) & 0xFF] << 24);
    uint32_t r3 = ((uint32_t)sb[(s3 >> 24) & 0xFF]) |
                  ((uint32_t)sb[(s0 >> 24) & 0xFF] << 8) |
                  ((uint32_t)sb[(s1 >> 24) & 0xFF] << 16) |
                  ((uint32_t)sb[(s2 >> 24) & 0xFF] << 24);

    r0 ^= rk[4*ROUNDS + 0];
    r1 ^= rk[4*ROUNDS + 1];
    r2 ^= rk[4*ROUNDS + 2];
    r3 ^= rk[4*ROUNDS + 3];

    reinterpret_cast<uint4*>(out)[idx] = make_uint4(r0,r1,r2,r3);
}

__global__ void aes_ecb_encrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    aes_ecb_encrypt_impl<10>(in, out, nBlocks);
}

__global__ void aes_ecb_encrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    aes_ecb_encrypt_impl<14>(in, out, nBlocks);
}

template<int ROUNDS>
__device__ void aes_ecb_decrypt_impl(const uint8_t* __restrict__ in,
                                     uint8_t* __restrict__ out,
                                     size_t nBlocks) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;

    const uint32_t *rk = d_roundKeys;
    const uint32_t *rk_last = rk + 4*ROUNDS;

    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = inBlock.x ^ rk_last[0];
    uint32_t s1 = inBlock.y ^ rk_last[1];
    uint32_t s2 = inBlock.z ^ rk_last[2];
    uint32_t s3 = inBlock.w ^ rk_last[3];

    uint32_t t0,t1,t2,t3;
#pragma unroll
    for (int r = ROUNDS-1; r >= 1; --r) {
        AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4*r);
        s0=t0; s1=t1; s2=t2; s3=t3;
    }

    const uint8_t *isbox = d_inv_sbox;
    uint32_t r0 = ((uint32_t)isbox[ s0        & 0xFF]) |
                  ((uint32_t)isbox[ s1        & 0xFF] << 8) |
                  ((uint32_t)isbox[ s2        & 0xFF] << 16) |
                  ((uint32_t)isbox[ s3        & 0xFF] << 24);
    uint32_t r1 = ((uint32_t)isbox[(s3 >>  8) & 0xFF]) |
                  ((uint32_t)isbox[(s0 >>  8) & 0xFF] << 8) |
                  ((uint32_t)isbox[(s1 >>  8) & 0xFF] << 16) |
                  ((uint32_t)isbox[(s2 >>  8) & 0xFF] << 24);
    uint32_t r2 = ((uint32_t)isbox[(s2 >> 16) & 0xFF]) |
                  ((uint32_t)isbox[(s3 >> 16) & 0xFF] << 8) |
                  ((uint32_t)isbox[(s0 >> 16) & 0xFF] << 16) |
                  ((uint32_t)isbox[(s1 >> 16) & 0xFF] << 24);
    uint32_t r3 = ((uint32_t)isbox[(s1 >> 24) & 0xFF]) |
                  ((uint32_t)isbox[(s2 >> 24) & 0xFF] << 8) |
                  ((uint32_t)isbox[(s3 >> 24) & 0xFF] << 16) |
                  ((uint32_t)isbox[(s0 >> 24) & 0xFF] << 24);

    r0 ^= rk[0]; r1 ^= rk[1]; r2 ^= rk[2]; r3 ^= rk[3];

    reinterpret_cast<uint4*>(out)[idx] = make_uint4(r0,r1,r2,r3);
}

__global__ void aes_ecb_decrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    aes_ecb_decrypt_impl<10>(in, out, nBlocks);
}

__global__ void aes_ecb_decrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks) {
    aes_ecb_decrypt_impl<14>(in, out, nBlocks);
}

// Explicit instantiation of the templated device implementations so that the
// linker resolves references from the wrapper kernels.
template __device__ void aes_ecb_encrypt_impl<10>(const uint8_t*, uint8_t*, size_t);
template __device__ void aes_ecb_encrypt_impl<14>(const uint8_t*, uint8_t*, size_t);
template __device__ void aes_ecb_decrypt_impl<10>(const uint8_t*, uint8_t*, size_t);
template __device__ void aes_ecb_decrypt_impl<14>(const uint8_t*, uint8_t*, size_t);

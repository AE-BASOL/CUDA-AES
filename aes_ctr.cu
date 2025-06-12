#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];

#define CTR_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk) do { \
    o0 = sh_T0[(s0)&0xFF] ^ sh_T1[((s1)>>8)&0xFF] ^ \
         sh_T2[((s2)>>16)&0xFF] ^ sh_T3[((s3)>>24)&0xFF] ^ (rk)[0]; \
    o1 = sh_T0[(s1)&0xFF] ^ sh_T1[((s2)>>8)&0xFF] ^ \
         sh_T2[((s3)>>16)&0xFF] ^ sh_T3[((s0)>>24)&0xFF] ^ (rk)[1]; \
    o2 = sh_T0[(s2)&0xFF] ^ sh_T1[((s3)>>8)&0xFF] ^ \
         sh_T2[((s0)>>16)&0xFF] ^ sh_T3[((s1)>>24)&0xFF] ^ (rk)[2]; \
    o3 = sh_T0[(s3)&0xFF] ^ sh_T1[((s0)>>8)&0xFF] ^ \
         sh_T2[((s1)>>16)&0xFF] ^ sh_T3[((s2)>>24)&0xFF] ^ (rk)[3]; \
} while(0)

template<int ROUNDS>
__device__ inline void ctr_keystream(uint64_t ctr_lo, uint64_t ctr_hi,
                                     uint32_t &k0, uint32_t &k1, uint32_t &k2, uint32_t &k3,
                                     const uint32_t *rk, const uint8_t *sb,
                                     uint32_t *sh_T0, uint32_t *sh_T1,
                                     uint32_t *sh_T2, uint32_t *sh_T3) {
    uint32_t s0 = (uint32_t)ctr_lo;
    uint32_t s1 = (uint32_t)(ctr_lo >> 32);
    uint32_t s2 = (uint32_t)ctr_hi;
    uint32_t s3 = (uint32_t)(ctr_hi >> 32);
    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];
    uint32_t t0,t1,t2,t3;
#pragma unroll
    for(int r=1; r<ROUNDS; ++r) {
        CTR_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4*r);
        s0=t0; s1=t1; s2=t2; s3=t3;
    }
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
    k0 ^= rk[4*ROUNDS + 0];
    k1 ^= rk[4*ROUNDS + 1];
    k2 ^= rk[4*ROUNDS + 2];
    k3 ^= rk[4*ROUNDS + 3];
}

template<int ROUNDS>
__device__ void aes_ctr_encrypt_impl(const uint8_t* __restrict__ in,
                                     uint8_t* __restrict__ out,
                                     size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
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
    const uint8_t *sb = sh_sbox;

    uint64_t ctr_lo = ctrLo + idx;
    uint64_t ctr_hi = ctrHi + (ctr_lo < ctrLo);

    uint32_t k0,k1,k2,k3;
    ctr_keystream<ROUNDS>(ctr_lo, ctr_hi, k0,k1,k2,k3, rk, sb, sh_T0,sh_T1,sh_T2,sh_T3);
    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint4 outBlock = make_uint4(inBlock.x ^ k0,
                                inBlock.y ^ k1,
                                inBlock.z ^ k2,
                                inBlock.w ^ k3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;

    size_t idx2 = idx + stride;
    if (idx2 < nBlocks) {
        ctr_lo = ctrLo + idx2;
        ctr_hi = ctrHi + (ctr_lo < ctrLo);
        ctr_keystream<ROUNDS>(ctr_lo, ctr_hi, k0,k1,k2,k3, rk, sb, sh_T0,sh_T1,sh_T2,sh_T3);
        inBlock = reinterpret_cast<const uint4*>(in)[idx2];
        outBlock = make_uint4(inBlock.x ^ k0,
                              inBlock.y ^ k1,
                              inBlock.z ^ k2,
                              inBlock.w ^ k3);
        reinterpret_cast<uint4*>(out)[idx2] = outBlock;
    }
}

__global__ void aes_ctr_encrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi) {
    aes_ctr_encrypt_impl<10>(in, out, nBlocks, ctrLo, ctrHi);
}

__global__ void aes_ctr_encrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi) {
    aes_ctr_encrypt_impl<14>(in, out, nBlocks, ctrLo, ctrHi);
}

template<int ROUNDS>
__device__ void aes_ctr_decrypt_impl(const uint8_t* __restrict__ in,
                                     uint8_t* __restrict__ out,
                                     size_t nBlocks, uint64_t ctrLo, uint64_t ctrHi) {
    aes_ctr_encrypt_impl<ROUNDS>(in, out, nBlocks, ctrLo, ctrHi);
}

__global__ void aes_ctr_decrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi) {
    aes_ctr_decrypt_impl<10>(in, out, nBlocks, ctrLo, ctrHi);
}

__global__ void aes_ctr_decrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi) {
    aes_ctr_decrypt_impl<14>(in, out, nBlocks, ctrLo, ctrHi);
}

// Explicit instantiation of device implementations used by the wrapper kernels
template __device__ void aes_ctr_encrypt_impl<10>(const uint8_t*, uint8_t*, size_t, uint64_t, uint64_t);
template __device__ void aes_ctr_encrypt_impl<14>(const uint8_t*, uint8_t*, size_t, uint64_t, uint64_t);
template __device__ void aes_ctr_decrypt_impl<10>(const uint8_t*, uint8_t*, size_t, uint64_t, uint64_t);
template __device__ void aes_ctr_decrypt_impl<14>(const uint8_t*, uint8_t*, size_t, uint64_t, uint64_t);

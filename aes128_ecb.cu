#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_sbox[256], d_inv_sbox[256];

#define ENC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk) \
    do { \
        o0 = sh_T0[(s0) & 0xFF] ^ sh_T1[((s1)>>8)&0xFF] ^ \
             sh_T2[((s2)>>16)&0xFF] ^ sh_T3[((s3)>>24)&0xFF] ^ (rk)[0]; \
        o1 = sh_T0[(s1) & 0xFF] ^ sh_T1[((s2)>>8)&0xFF] ^ \
             sh_T2[((s3)>>16)&0xFF] ^ sh_T3[((s0)>>24)&0xFF] ^ (rk)[1]; \
        o2 = sh_T0[(s2) & 0xFF] ^ sh_T1[((s3)>>8)&0xFF] ^ \
             sh_T2[((s0)>>16)&0xFF] ^ sh_T3[((s1)>>24)&0xFF] ^ (rk)[2]; \
        o3 = sh_T0[(s3) & 0xFF] ^ sh_T1[((s0)>>8)&0xFF] ^ \
             sh_T2[((s1)>>16)&0xFF] ^ sh_T3[((s2)>>24)&0xFF] ^ (rk)[3]; \
    } while(0)

#define DEC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk) \
    do { \
        o0 = sh_U0[(s0) & 0xFF] ^ sh_U1[((s3)>>8)&0xFF] ^ \
             sh_U2[((s2)>>16)&0xFF] ^ sh_U3[((s1)>>24)&0xFF] ^ (rk)[0]; \
        o1 = sh_U0[(s1) & 0xFF] ^ sh_U1[((s0)>>8)&0xFF] ^ \
             sh_U2[((s3)>>16)&0xFF] ^ sh_U3[((s2)>>24)&0xFF] ^ (rk)[1]; \
        o2 = sh_U0[(s2) & 0xFF] ^ sh_U1[((s1)>>8)&0xFF] ^ \
             sh_U2[((s0)>>16)&0xFF] ^ sh_U3[((s3)>>24)&0xFF] ^ (rk)[2]; \
        o3 = sh_U0[(s3) & 0xFF] ^ sh_U1[((s2)>>8)&0xFF] ^ \
             sh_U2[((s1)>>16)&0xFF] ^ sh_U3[((s0)>>24)&0xFF] ^ (rk)[3]; \
    } while(0)

__global__ void aes128_ecb_encrypt(const uint8_t* in, uint8_t* out, size_t nBlocks) {
    const size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
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
    size_t i = idx;
    if (i < nBlocks) {
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0 = block.x ^ rk[0];
        uint32_t s1 = block.y ^ rk[1];
        uint32_t s2 = block.z ^ rk[2];
        uint32_t s3 = block.w ^ rk[3];
        uint32_t t0,t1,t2,t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4);  s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8);  s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;
        const uint8_t* sb = sh_sbox;
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
        reinterpret_cast<uint4*>(out)[i] = make_uint4(k0,k1,k2,k3);
    }
    i += stride;
    if (i < nBlocks) {
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0 = block.x ^ rk[0];
        uint32_t s1 = block.y ^ rk[1];
        uint32_t s2 = block.z ^ rk[2];
        uint32_t s3 = block.w ^ rk[3];
        uint32_t t0,t1,t2,t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4);  s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8);  s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
        ENC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;
        const uint8_t* sb = sh_sbox;
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
        reinterpret_cast<uint4*>(out)[i] = make_uint4(k0,k1,k2,k3);
    }
}

__global__ void aes128_ecb_decrypt(const uint8_t* in, uint8_t* out, size_t nBlocks) {
    const size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    if (idx >= nBlocks) return;
    __shared__ uint32_t sh_U0[256], sh_U1[256], sh_U2[256], sh_U3[256];
    __shared__ uint8_t  sh_inv[256];
    if (threadIdx.x < 256) {
        sh_U0[threadIdx.x] = d_U0[threadIdx.x];
        sh_U1[threadIdx.x] = d_U1[threadIdx.x];
        sh_U2[threadIdx.x] = d_U2[threadIdx.x];
        sh_U3[threadIdx.x] = d_U3[threadIdx.x];
        sh_inv[threadIdx.x] = d_inv_sbox[threadIdx.x];
    }
    __syncthreads();
    const uint32_t *rk = d_roundKeys;
    size_t i = idx;
    if (i < nBlocks) {
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0 = block.x ^ rk[40];
        uint32_t s1 = block.y ^ rk[41];
        uint32_t s2 = block.z ^ rk[42];
        uint32_t s3 = block.w ^ rk[43];
        uint32_t t0,t1,t2,t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8);  s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4);  s0=t0; s1=t1; s2=t2; s3=t3;
        const uint8_t* isb = sh_inv;
        uint32_t k0 = ((uint32_t)isb[ s0        & 0xFF]) |
                      ((uint32_t)isb[(s3 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s2 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s1 >> 24) & 0xFF] << 24);
        uint32_t k1 = ((uint32_t)isb[ s1        & 0xFF]) |
                      ((uint32_t)isb[(s0 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s3 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s2 >> 24) & 0xFF] << 24);
        uint32_t k2 = ((uint32_t)isb[ s2        & 0xFF]) |
                      ((uint32_t)isb[(s1 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s0 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s3 >> 24) & 0xFF] << 24);
        uint32_t k3 = ((uint32_t)isb[ s3        & 0xFF]) |
                      ((uint32_t)isb[(s2 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s1 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s0 >> 24) & 0xFF] << 24);
        k0 ^= rk[0]; k1 ^= rk[1]; k2 ^= rk[2]; k3 ^= rk[3];
        reinterpret_cast<uint4*>(out)[i] = make_uint4(k0,k1,k2,k3);
    }
    i += stride;
    if (i < nBlocks) {
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0 = block.x ^ rk[40];
        uint32_t s1 = block.y ^ rk[41];
        uint32_t s2 = block.z ^ rk[42];
        uint32_t s3 = block.w ^ rk[43];
        uint32_t t0,t1,t2,t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8);  s0=t0; s1=t1; s2=t2; s3=t3;
        DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4);  s0=t0; s1=t1; s2=t2; s3=t3;
        const uint8_t* isb = sh_inv;
        uint32_t k0 = ((uint32_t)isb[ s0        & 0xFF]) |
                      ((uint32_t)isb[(s3 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s2 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s1 >> 24) & 0xFF] << 24);
        uint32_t k1 = ((uint32_t)isb[ s1        & 0xFF]) |
                      ((uint32_t)isb[(s0 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s3 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s2 >> 24) & 0xFF] << 24);
        uint32_t k2 = ((uint32_t)isb[ s2        & 0xFF]) |
                      ((uint32_t)isb[(s1 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s0 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s3 >> 24) & 0xFF] << 24);
        uint32_t k3 = ((uint32_t)isb[ s3        & 0xFF]) |
                      ((uint32_t)isb[(s2 >>  8) & 0xFF] << 8) |
                      ((uint32_t)isb[(s1 >> 16) & 0xFF] << 16) |
                      ((uint32_t)isb[(s0 >> 24) & 0xFF] << 24);
        k0 ^= rk[0]; k1 ^= rk[1]; k2 ^= rk[2]; k3 ^= rk[3];
        reinterpret_cast<uint4*>(out)[i] = make_uint4(k0,k1,k2,k3);
    }
}

#undef ENC_ROUND
#undef DEC_ROUND

#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];

#define B0(x) (((x) >> 24) & 0xFF)
#define B1(x) (((x) >> 16) & 0xFF)
#define B2(x) (((x) >>  8) & 0xFF)
#define B3(x) ((x) & 0xFF)

#define AES_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                          \
    do {                                                               \
        o0 = sh_T0[B0(s0)] ^ sh_T1[B1(s1)] ^                           \
             sh_T2[B2(s2)] ^ sh_T3[B3(s3)] ^ (rk)[0];                  \
        o1 = sh_T0[B0(s1)] ^ sh_T1[B1(s2)] ^                           \
             sh_T2[B2(s3)] ^ sh_T3[B3(s0)] ^ (rk)[1];                  \
        o2 = sh_T0[B0(s2)] ^ sh_T1[B1(s3)] ^                           \
             sh_T2[B2(s0)] ^ sh_T3[B3(s1)] ^ (rk)[2];                  \
        o3 = sh_T0[B0(s3)] ^ sh_T1[B1(s0)] ^                           \
             sh_T2[B2(s1)] ^ sh_T3[B3(s2)] ^ (rk)[3];                  \
    } while (0)

#define AES_DEC_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                      \
    do {                                                               \
        o0 = sh_U0[B0(s0)] ^ sh_U1[B1(s3)] ^                           \
             sh_U2[B2(s2)] ^ sh_U3[B3(s1)] ^ (rk)[0];                  \
        o1 = sh_U0[B0(s1)] ^ sh_U1[B1(s0)] ^                           \
             sh_U2[B2(s3)] ^ sh_U3[B3(s2)] ^ (rk)[1];                  \
        o2 = sh_U0[B0(s2)] ^ sh_U1[B1(s1)] ^                           \
             sh_U2[B2(s0)] ^ sh_U3[B3(s3)] ^ (rk)[2];                  \
        o3 = sh_U0[B0(s3)] ^ sh_U1[B1(s2)] ^                           \
             sh_U2[B2(s1)] ^ sh_U3[B3(s0)] ^ (rk)[3];                  \
    } while (0)

__global__ void aes128_ecb_encrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks)
{
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

    const uint32_t *rk = d_roundKeys;
    uint4 blk = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = blk.x ^ rk[0];
    uint32_t s1 = blk.y ^ rk[1];
    uint32_t s2 = blk.z ^ rk[2];
    uint32_t s3 = blk.w ^ rk[3];

    uint32_t t0,t1,t2,t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4);  s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8);  s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;

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

    r0 ^= rk[40]; r1 ^= rk[41]; r2 ^= rk[42]; r3 ^= rk[43];
    reinterpret_cast<uint4*>(out)[idx] = make_uint4(r0,r1,r2,r3);
}

__global__ void aes128_ecb_decrypt(const uint8_t *__restrict__ in,
                                   uint8_t *__restrict__ out,
                                   size_t nBlocks)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;

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

    const uint32_t *rk = d_roundKeys;
    const uint32_t *rk_last = rk + 40;

    uint4 blk = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = blk.x ^ rk_last[0];
    uint32_t s1 = blk.y ^ rk_last[1];
    uint32_t s2 = blk.z ^ rk_last[2];
    uint32_t s3 = blk.w ^ rk_last[3];

    uint32_t t0,t1,t2,t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+36); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+32); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+28); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+24); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+20); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+16); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+12); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+8 ); s0=t0; s1=t1; s2=t2; s3=t3;
    AES_DEC_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk+4 ); s0=t0; s1=t1; s2=t2; s3=t3;

    const uint8_t *isbox = sh_isbox;
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

    r0 ^= rk[0]; r1 ^= rk[1]; r2 ^= rk[2]; r3 ^= rk[3];
    reinterpret_cast<uint4*>(out)[idx] = make_uint4(r0,r1,r2,r3);
}


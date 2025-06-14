/********************************************************************
 *  aes128_ecb.cu  – one-thread-per-block AES-128 ECB encryptor
 *                   **little-endian state + tables**
 *******************************************************************/
#include <cuda_runtime.h>
#include <cstdint>

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256],
                                       d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];


/* helpers */
#define B0(x) (((x) >> 24) & 0xFF)
#define B1(x) (((x) >> 16) & 0xFF)
#define B2(x) (((x) >>  8) & 0xFF)
#define B3(x) ((x) & 0xFF)

/* corrected T-table round (matches OpenSSL te0-te3) */
#define AES_ROUND(o0,o1,o2,o3,s0,s1,s2,s3,rk)                    \
do {                                                            \
    (o0)=sh_T0[B0(s0)] ^ sh_T1[B1(s1)] ^                        \
         sh_T2[B2(s2)] ^ sh_T3[B3(s3)] ^ (rk)[0];               \
    (o1)=sh_T0[B0(s1)] ^ sh_T1[B1(s2)] ^                        \
         sh_T2[B2(s3)] ^ sh_T3[B3(s0)] ^ (rk)[1];               \
    (o2)=sh_T0[B0(s2)] ^ sh_T1[B1(s3)] ^                        \
         sh_T2[B2(s0)] ^ sh_T3[B3(s1)] ^ (rk)[2];               \
    (o3)=sh_T0[B0(s3)] ^ sh_T1[B1(s0)] ^                        \
         sh_T2[B2(s1)] ^ sh_T3[B3(s2)] ^ (rk)[3];               \
} while (0)

/* ====================== kernel ============================== */
__global__ void aes128_ecb_encrypt(const uint8_t *in,
                                   uint8_t       *out,
                                   size_t         nBlocks)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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

    const uint32_t *rk = d_roundKeys;         // 44 words

    /* ---- load 16 input bytes as 4 *little-endian* words ---- */
    uint4 block = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = block.x;
    uint32_t s1 = block.y;
    uint32_t s2 = block.z;
    uint32_t s3 = block.w;

    /* initial AddRoundKey */
    s0 ^= rk[0];  s1 ^= rk[1];
    s2 ^= rk[2];  s3 ^= rk[3];

    /* 9 full rounds */
    uint32_t t0,t1,t2,t3;
#pragma unroll
    for (int r = 1; r <= 9; ++r) {
        AES_ROUND(t0,t1,t2,t3, s0,s1,s2,s3, rk + 4*r);
        s0=t0; s1=t1; s2=t2; s3=t3;
    }

    /* final round (SubBytes + ShiftRows) */
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

    /* final AddRoundKey */
    r0 ^= rk[40];
    r1 ^= rk[41];
    r2 ^= rk[42];
    r3 ^= rk[43];

    uint4 outBlock = make_uint4(r0, r1, r2, r3);
    reinterpret_cast<uint4*>(out)[idx] = outBlock;
}

__global__ void aes128_ecb_decrypt(const uint8_t *in,
                                   uint8_t       *out,
                                   size_t         nBlocks)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBlocks) return;

    const uint32_t *rk = d_roundKeys;         // 44 words for AES-128
    const uint32_t *rk_last = rk + 40;        // last round key for AES-128

    // Load ciphertext and initial AddRoundKey with last round key
    uint4 inBlock = reinterpret_cast<const uint4*>(in)[idx];
    uint32_t s0 = inBlock.x ^ rk_last[0];
    uint32_t s1 = inBlock.y ^ rk_last[1];
    uint32_t s2 = inBlock.z ^ rk_last[2];
    uint32_t s3 = inBlock.w ^ rk_last[3];

    // 9 full inverse rounds (use U-tables, as AES-256)
    uint32_t t0, t1, t2, t3;
#pragma unroll
    for (int r = 9; r >= 1; --r) {
        t0 = d_U0[(s0      ) & 0xFF] ^ d_U1[(s3 >>  8) & 0xFF] ^
             d_U2[(s2 >> 16) & 0xFF] ^ d_U3[(s1 >> 24) & 0xFF] ^ rk[4*r + 0];
        t1 = d_U0[(s1      ) & 0xFF] ^ d_U1[(s0 >>  8) & 0xFF] ^
             d_U2[(s3 >> 16) & 0xFF] ^ d_U3[(s2 >> 24) & 0xFF] ^ rk[4*r + 1];
        t2 = d_U0[(s2      ) & 0xFF] ^ d_U1[(s1 >>  8) & 0xFF] ^
             d_U2[(s0 >> 16) & 0xFF] ^ d_U3[(s3 >> 24) & 0xFF] ^ rk[4*r + 2];
        t3 = d_U0[(s3      ) & 0xFF] ^ d_U1[(s2 >>  8) & 0xFF] ^
             d_U2[(s1 >> 16) & 0xFF] ^ d_U3[(s0 >> 24) & 0xFF] ^ rk[4*r + 3];
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


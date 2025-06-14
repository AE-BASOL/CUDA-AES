#include "aes_common.h"

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];

__device__ __forceinline__ uint8_t gf_xtime(uint8_t x){ return static_cast<uint8_t>((x<<1) ^ ((x&0x80)?0x1B:0)); }
__device__ __forceinline__ uint8_t gf_mul2(uint8_t x){ return gf_xtime(x); }
__device__ __forceinline__ uint8_t gf_mul4(uint8_t x){ return gf_mul2(gf_mul2(x)); }
__device__ __forceinline__ uint8_t gf_mul8(uint8_t x){ return gf_mul2(gf_mul4(x)); }
__device__ __forceinline__ uint8_t gf_mul3(uint8_t x){ return gf_mul2(x) ^ x; }
__device__ __forceinline__ uint8_t gf_mul9(uint8_t x){ return gf_mul8(x) ^ x; }
__device__ __forceinline__ uint8_t gf_mul11(uint8_t x){ return gf_mul8(x) ^ gf_mul2(x) ^ x; }
__device__ __forceinline__ uint8_t gf_mul13(uint8_t x){ return gf_mul8(x) ^ gf_mul4(x) ^ x; }
__device__ __forceinline__ uint8_t gf_mul14(uint8_t x){ return gf_mul8(x) ^ gf_mul4(x) ^ gf_mul2(x); }

__device__ __forceinline__ uint32_t inv_mix_columns_word(uint32_t w){
    uint8_t b0=w&0xFF, b1=(w>>8)&0xFF, b2=(w>>16)&0xFF, b3=(w>>24)&0xFF;
    uint8_t r0 = gf_mul14(b0) ^ gf_mul11(b1) ^ gf_mul13(b2) ^ gf_mul9(b3);
    uint8_t r1 = gf_mul9(b0)  ^ gf_mul14(b1) ^ gf_mul11(b2) ^ gf_mul13(b3);
    uint8_t r2 = gf_mul13(b0) ^ gf_mul9(b1)  ^ gf_mul14(b2) ^ gf_mul11(b3);
    uint8_t r3 = gf_mul11(b0) ^ gf_mul13(b1) ^ gf_mul9(b2)  ^ gf_mul14(b3);
    return (uint32_t)r0 | ((uint32_t)r1<<8) | ((uint32_t)r2<<16) | ((uint32_t)r3<<24);
}

__global__ void aes256_ecb_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks){
    const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x*gridDim.x;
    if(idx>=nBlocks) return;
    __shared__ uint32_t sh_T0[256], sh_T1[256], sh_T2[256], sh_T3[256];
    __shared__ uint8_t  sh_sbox[256];
    if(threadIdx.x<256){
        sh_T0[threadIdx.x]=d_T0[threadIdx.x];
        sh_T1[threadIdx.x]=d_T1[threadIdx.x];
        sh_T2[threadIdx.x]=d_T2[threadIdx.x];
        sh_T3[threadIdx.x]=d_T3[threadIdx.x];
        sh_sbox[threadIdx.x]=d_sbox[threadIdx.x];
    }
    __syncthreads();
    const uint32_t* rk=d_roundKeys;
    size_t i=idx;
    for(int iter=0; iter<2; ++iter, i+=stride){
        if(i>=nBlocks) break;
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0=block.x^rk[0];
        uint32_t s1=block.y^rk[1];
        uint32_t s2=block.z^rk[2];
        uint32_t s3=block.w^rk[3];
        uint32_t t0,t1,t2,t3;
#pragma unroll
        for(int r=1;r<=13;++r){
            t0 = sh_T0[s0&0xFF]^sh_T1[(s1>>8)&0xFF]^sh_T2[(s2>>16)&0xFF]^sh_T3[(s3>>24)&0xFF]^rk[4*r+0];
            t1 = sh_T0[s1&0xFF]^sh_T1[(s2>>8)&0xFF]^sh_T2[(s3>>16)&0xFF]^sh_T3[(s0>>24)&0xFF]^rk[4*r+1];
            t2 = sh_T0[s2&0xFF]^sh_T1[(s3>>8)&0xFF]^sh_T2[(s0>>16)&0xFF]^sh_T3[(s1>>24)&0xFF]^rk[4*r+2];
            t3 = sh_T0[s3&0xFF]^sh_T1[(s0>>8)&0xFF]^sh_T2[(s1>>16)&0xFF]^sh_T3[(s2>>24)&0xFF]^rk[4*r+3];
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        const uint8_t* sb=sh_sbox;
        uint32_t o0=(uint32_t)sb[s0&0xFF]|((uint32_t)sb[(s1>>8)&0xFF]<<8)|((uint32_t)sb[(s2>>16)&0xFF]<<16)|((uint32_t)sb[(s3>>24)&0xFF]<<24);
        uint32_t o1=(uint32_t)sb[s1&0xFF]|((uint32_t)sb[(s2>>8)&0xFF]<<8)|((uint32_t)sb[(s3>>16)&0xFF]<<16)|((uint32_t)sb[(s0>>24)&0xFF]<<24);
        uint32_t o2=(uint32_t)sb[s2&0xFF]|((uint32_t)sb[(s3>>8)&0xFF]<<8)|((uint32_t)sb[(s0>>16)&0xFF]<<16)|((uint32_t)sb[(s1>>24)&0xFF]<<24);
        uint32_t o3=(uint32_t)sb[s3&0xFF]|((uint32_t)sb[(s0>>8)&0xFF]<<8)|((uint32_t)sb[(s1>>16)&0xFF]<<16)|((uint32_t)sb[(s2>>24)&0xFF]<<24);
        o0^=rk[56]; o1^=rk[57]; o2^=rk[58]; o3^=rk[59];
        reinterpret_cast<uint4*>(out)[i]=make_uint4(o0,o1,o2,o3);
    }
}

__global__ void aes256_ecb_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks){
    const size_t idx=blockIdx.x*blockDim.x+threadIdx.x;
    const size_t stride=blockDim.x*gridDim.x;
    if(idx>=nBlocks) return;
    __shared__ uint32_t sh_U0[256], sh_U1[256], sh_U2[256], sh_U3[256];
    __shared__ uint8_t  sh_inv[256];
    if(threadIdx.x<256){
        sh_U0[threadIdx.x]=d_U0[threadIdx.x];
        sh_U1[threadIdx.x]=d_U1[threadIdx.x];
        sh_U2[threadIdx.x]=d_U2[threadIdx.x];
        sh_U3[threadIdx.x]=d_U3[threadIdx.x];
        sh_inv[threadIdx.x]=d_inv_sbox[threadIdx.x];
    }
    __syncthreads();
    const uint32_t* rk=d_roundKeys;
    size_t i=idx;
    for(int iter=0; iter<2; ++iter, i+=stride){
        if(i>=nBlocks) break;
        uint4 block = reinterpret_cast<const uint4*>(in)[i];
        uint32_t s0=block.x^rk[56];
        uint32_t s1=block.y^rk[57];
        uint32_t s2=block.z^rk[58];
        uint32_t s3=block.w^rk[59];
        uint32_t t0,t1,t2,t3;
        for(int r=13;r>=1;--r){
            uint32_t k0=inv_mix_columns_word(rk[4*(r-1)+0]);
            uint32_t k1=inv_mix_columns_word(rk[4*(r-1)+1]);
            uint32_t k2=inv_mix_columns_word(rk[4*(r-1)+2]);
            uint32_t k3=inv_mix_columns_word(rk[4*(r-1)+3]);
            t0=sh_U0[s0&0xFF]^sh_U1[(s3>>8)&0xFF]^sh_U2[(s2>>16)&0xFF]^sh_U3[(s1>>24)&0xFF]^k0;
            t1=sh_U0[s1&0xFF]^sh_U1[(s0>>8)&0xFF]^sh_U2[(s3>>16)&0xFF]^sh_U3[(s2>>24)&0xFF]^k1;
            t2=sh_U0[s2&0xFF]^sh_U1[(s1>>8)&0xFF]^sh_U2[(s0>>16)&0xFF]^sh_U3[(s3>>24)&0xFF]^k2;
            t3=sh_U0[s3&0xFF]^sh_U1[(s2>>8)&0xFF]^sh_U2[(s1>>16)&0xFF]^sh_U3[(s0>>24)&0xFF]^k3;
            s0=t0; s1=t1; s2=t2; s3=t3;
        }
        const uint8_t* is=sh_inv;
        uint32_t o0=(uint32_t)is[s0&0xFF]|((uint32_t)is[(s3>>8)&0xFF]<<8)|((uint32_t)is[(s2>>16)&0xFF]<<16)|((uint32_t)is[(s1>>24)&0xFF]<<24);
        uint32_t o1=(uint32_t)is[s1&0xFF]|((uint32_t)is[(s0>>8)&0xFF]<<8)|((uint32_t)is[(s3>>16)&0xFF]<<16)|((uint32_t)is[(s2>>24)&0xFF]<<24);
        uint32_t o2=(uint32_t)is[s2&0xFF]|((uint32_t)is[(s1>>8)&0xFF]<<8)|((uint32_t)is[(s0>>16)&0xFF]<<16)|((uint32_t)is[(s3>>24)&0xFF]<<24);
        uint32_t o3=(uint32_t)is[s3&0xFF]|((uint32_t)is[(s2>>8)&0xFF]<<8)|((uint32_t)is[(s1>>16)&0xFF]<<16)|((uint32_t)is[(s0>>24)&0xFF]<<24);
        o0^=rk[0]; o1^=rk[1]; o2^=rk[2]; o3^=rk[3];
        reinterpret_cast<uint4*>(out)[i]=make_uint4(o0,o1,o2,o3);
    }
}


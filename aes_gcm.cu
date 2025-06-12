#include "aes_common.h"
#include <stdint.h>

extern __device__ __constant__ uint32_t d_roundKeys[60];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint64_t d_H_pow_hi[32];
extern __device__ __constant__ uint64_t d_H_pow_lo[32];

static __device__ inline void gf_mul128(uint64_t &Ah, uint64_t &Al,
                                        uint64_t Bh, uint64_t Bl) {
    uint64_t Zh = 0ull, Zl = 0ull;
    uint64_t Vh = Bh, Vl = Bl;
    const uint64_t R = 0xE100000000000000ULL;
    for (int i=0;i<128;++i) {
        if (Al & 1ULL) { Zl ^= Vl; Zh ^= Vh; }
        bool carry = (Vl & 1ULL);
        Vl = (Vl >> 1) | (Vh << 63);
        Vh >>= 1;
        if (carry) Vh ^= R;
        Al = (Al >> 1) | (Ah << 63);
        Ah >>= 1;
    }
    Ah = Zh; Al = Zl;
}

#define CTR_STEP(o0,o1,o2,o3,s0,s1,s2,s3,rk) do { \
    o0 = d_T0[s0 & 0xFF] ^ d_T1[(s1>>8)&0xFF] ^ d_T2[(s2>>16)&0xFF] ^ d_T3[(s3>>24)&0xFF] ^ (rk)[0]; \
    o1 = d_T0[s1 & 0xFF] ^ d_T1[(s2>>8)&0xFF] ^ d_T2[(s3>>16)&0xFF] ^ d_T3[(s0>>24)&0xFF] ^ (rk)[1]; \
    o2 = d_T0[s2 & 0xFF] ^ d_T1[(s3>>8)&0xFF] ^ d_T2[(s0>>16)&0xFF] ^ d_T3[(s1>>24)&0xFF] ^ (rk)[2]; \
    o3 = d_T0[s3 & 0xFF] ^ d_T1[(s0>>8)&0xFF] ^ d_T2[(s1>>16)&0xFF] ^ d_T3[(s2>>24)&0xFF] ^ (rk)[3]; \
} while(0)

template<int ROUNDS>
__device__ void ctr_keystream_gcm(uint64_t ctr_lo, uint64_t ctr_hi, uint8_t ks[16]) {
    uint32_t s0=(uint32_t)ctr_lo, s1=(uint32_t)(ctr_lo>>32);
    uint32_t s2=(uint32_t)ctr_hi, s3=(uint32_t)(ctr_hi>>32);
    const uint32_t *rk = d_roundKeys;
    s0^=rk[0]; s1^=rk[1]; s2^=rk[2]; s3^=rk[3];
    uint32_t t0,t1,t2,t3;
#pragma unroll
    for(int r=1;r<ROUNDS;++r){
        CTR_STEP(t0,t1,t2,t3,s0,s1,s2,s3,rk+4*r);
        s0=t0; s1=t1; s2=t2; s3=t3;
    }
    const uint8_t *sb=d_sbox;
    ks[0]=sb[s0 & 0xFF];      ks[4]=sb[(s1>>8)&0xFF];
    ks[8]=sb[(s2>>16)&0xFF]; ks[12]=sb[(s3>>24)&0xFF];
    ks[1]=sb[s1 & 0xFF];      ks[5]=sb[(s2>>8)&0xFF];
    ks[9]=sb[(s3>>16)&0xFF];  ks[13]=sb[(s0>>24)&0xFF];
    ks[2]=sb[s2 & 0xFF];      ks[6]=sb[(s3>>8)&0xFF];
    ks[10]=sb[(s0>>16)&0xFF]; ks[14]=sb[(s1>>24)&0xFF];
    ks[3]=sb[s3 & 0xFF];      ks[7]=sb[(s0>>8)&0xFF];
    ks[11]=sb[(s1>>16)&0xFF]; ks[15]=sb[(s2>>24)&0xFF];
    ((uint32_t*)ks)[0]^=rk[4*ROUNDS+0];
    ((uint32_t*)ks)[1]^=rk[4*ROUNDS+1];
    ((uint32_t*)ks)[2]^=rk[4*ROUNDS+2];
    ((uint32_t*)ks)[3]^=rk[4*ROUNDS+3];
}

template<int ROUNDS>
__global__ void aes_gcm_encrypt(const uint8_t* __restrict__ plain,
                                uint8_t* __restrict__ cipher,
                                size_t nBlocks,
                                const uint8_t* __restrict__ iv,
                                uint8_t* __restrict__ tagOut) {
    uint64_t IV_lo=0, IV_hi=0;
    if(threadIdx.x==0){
        uint32_t w0=0,w1=0,w2=0; memcpy(&w0,iv,4); memcpy(&w1,iv+4,4); memcpy(&w2,iv+8,4);
        uint32_t w3=0x01000000u; IV_lo=(uint64_t)w0|((uint64_t)w1<<32); IV_hi=(uint64_t)w2|((uint64_t)w3<<32);
    }
    __syncthreads();
    IV_lo=__shfl_sync(0xFFFFFFFF,IV_lo,0); IV_hi=__shfl_sync(0xFFFFFFFF,IV_hi,0);

    for(size_t i=threadIdx.x;i<nBlocks;i+=blockDim.x){
        uint64_t ctr_lo=IV_lo+i, ctr_hi=IV_hi+(ctr_lo<IV_lo);
        uint8_t ks[16];
        ctr_keystream_gcm<ROUNDS>(ctr_lo, ctr_hi, ks);
        const uint8_t *pt=plain + i*16; uint8_t *ct=cipher + i*16;
        ((uint32_t*)ct)[0]=((const uint32_t*)pt)[0]^((uint32_t*)ks)[0];
        ((uint32_t*)ct)[1]=((const uint32_t*)pt)[1]^((uint32_t*)ks)[1];
        ((uint32_t*)ct)[2]=((const uint32_t*)pt)[2]^((uint32_t*)ks)[2];
        ((uint32_t*)ct)[3]=((const uint32_t*)pt)[3]^((uint32_t*)ks)[3];
    }
    __syncthreads();

    uint32_t tid=threadIdx.x;
    if(tid<32){
        uint64_t step_hi=0, step_lo=1;
        for(int b=0;b<32;++b) if(32u&(1u<<b)) gf_mul128(step_hi,step_lo,d_H_pow_hi[b],d_H_pow_lo[b]);
        uint64_t pow_hi=0,pow_lo=1; uint32_t exp=(uint32_t)(nBlocks-1-tid);
        for(int b=0;b<32;++b) if(exp&(1u<<b)) gf_mul128(pow_hi,pow_lo,d_H_pow_hi[b],d_H_pow_lo[b]);
        uint64_t accum_hi=0, accum_lo=0;
        for(size_t j=tid;j<nBlocks;j+=32){
            uint64_t c_lo=((const uint64_t*)cipher)[2*j];
            uint64_t c_hi=((const uint64_t*)cipher)[2*j+1];
            uint64_t tmp_hi=c_hi,tmp_lo=c_lo;
            gf_mul128(tmp_hi,tmp_lo,pow_hi,pow_lo);
            accum_hi^=tmp_hi; accum_lo^=tmp_lo;
            gf_mul128(pow_hi,pow_lo,step_hi,step_lo);
        }
        for(int off=16;off>0;off>>=1){
            accum_hi^=__shfl_xor_sync(0xFFFFFFFF,accum_hi,off);
            accum_lo^=__shfl_xor_sync(0xFFFFFFFF,accum_lo,off);
        }
        if(tid==0){
            uint64_t lenBlock_lo=(uint64_t)nBlocks*16ull*8ull;
            uint64_t lenBlock_hi=0ull;
            accum_lo^=lenBlock_lo; accum_hi^=lenBlock_hi;
            gf_mul128(accum_hi,accum_lo,d_H_pow_hi[0],d_H_pow_lo[0]);
            ((uint64_t*)tagOut)[0]=accum_lo;
            ((uint64_t*)tagOut)[1]=accum_hi;
        }
    }
}

template<int ROUNDS>
__global__ void aes_gcm_decrypt(const uint8_t* __restrict__ cipher,
                                uint8_t* __restrict__ plain,
                                size_t nBlocks,
                                const uint8_t* __restrict__ iv,
                                const uint8_t* __restrict__ tag,
                                uint8_t* __restrict__ tagOut) {
    (void)tag; // tag verification is host-side
    uint64_t IV_lo=0, IV_hi=0;
    if(threadIdx.x==0){
        uint32_t w0=0,w1=0,w2=0; memcpy(&w0,iv,4); memcpy(&w1,iv+4,4); memcpy(&w2,iv+8,4);
        uint32_t w3=0x01000000u; IV_lo=(uint64_t)w0|((uint64_t)w1<<32); IV_hi=(uint64_t)w2|((uint64_t)w3<<32);
    }
    __syncthreads();
    IV_lo=__shfl_sync(0xFFFFFFFF,IV_lo,0); IV_hi=__shfl_sync(0xFFFFFFFF,IV_hi,0);

    for(size_t i=threadIdx.x;i<nBlocks;i+=blockDim.x){
        uint64_t ctr_lo=IV_lo+i, ctr_hi=IV_hi+(ctr_lo<IV_lo);
        uint8_t ks[16];
        ctr_keystream_gcm<ROUNDS>(ctr_lo, ctr_hi, ks);
        const uint8_t *ct=cipher + i*16; uint8_t *pt=plain + i*16;
        ((uint32_t*)pt)[0]=((const uint32_t*)ct)[0]^((uint32_t*)ks)[0];
        ((uint32_t*)pt)[1]=((const uint32_t*)ct)[1]^((uint32_t*)ks)[1];
        ((uint32_t*)pt)[2]=((const uint32_t*)ct)[2]^((uint32_t*)ks)[2];
        ((uint32_t*)pt)[3]=((const uint32_t*)ct)[3]^((uint32_t*)ks)[3];
    }
    __syncthreads();

    uint32_t tid=threadIdx.x;
    if(tid<32){
        uint64_t step_hi=0, step_lo=1;
        for(int b=0;b<32;++b) if(32u&(1u<<b)) gf_mul128(step_hi,step_lo,d_H_pow_hi[b],d_H_pow_lo[b]);
        uint64_t pow_hi=0,pow_lo=1; uint32_t exp=(uint32_t)(nBlocks-1-tid);
        for(int b=0;b<32;++b) if(exp&(1u<<b)) gf_mul128(pow_hi,pow_lo,d_H_pow_hi[b],d_H_pow_lo[b]);
        uint64_t accum_hi=0, accum_lo=0;
        for(size_t j=tid;j<nBlocks;j+=32){
            uint64_t c_lo=((const uint64_t*)cipher)[2*j];
            uint64_t c_hi=((const uint64_t*)cipher)[2*j+1];
            uint64_t tmp_hi=c_hi,tmp_lo=c_lo;
            gf_mul128(tmp_hi,tmp_lo,pow_hi,pow_lo);
            accum_hi^=tmp_hi; accum_lo^=tmp_lo;
            gf_mul128(pow_hi,pow_lo,step_hi,step_lo);
        }
        for(int off=16;off>0;off>>=1){
            accum_hi^=__shfl_xor_sync(0xFFFFFFFF,accum_hi,off);
            accum_lo^=__shfl_xor_sync(0xFFFFFFFF,accum_lo,off);
        }
        if(tid==0){
            uint64_t lenBlock_lo=(uint64_t)nBlocks*16ull*8ull;
            uint64_t lenBlock_hi=0ull;
            accum_lo^=lenBlock_lo; accum_hi^=lenBlock_hi;
            gf_mul128(accum_hi,accum_lo,d_H_pow_hi[0],d_H_pow_lo[0]);
            ((uint64_t*)tagOut)[0]=accum_lo;
            ((uint64_t*)tagOut)[1]=accum_hi;
        }
    }
}

#ifndef AES_COMMON_H
#define AES_COMMON_H

#include <cuda_runtime.h>
#include <cstdint>

// ----------------------------------------------------------
// Device constant memory declarations (defined in aes_tables.cu)
// These constants are declared extern here for use across modules
// and defined once in the aes_tables.cu implementation to avoid
// multiple definitions and linker conflicts.
// ----------------------------------------------------------
extern __device__ __constant__ uint8_t  d_sbox[256];
extern __device__ __constant__ uint8_t  d_inv_sbox[256];
extern __device__ __constant__ uint32_t d_T0[256], d_T1[256], d_T2[256], d_T3[256];
extern __device__ __constant__ uint32_t d_U0[256], d_U1[256], d_U2[256], d_U3[256];
extern __device__ __constant__ uint32_t d_roundKeys[60];  // supports AES-256 (max 60 words)
extern __device__ __constant__ uint64_t d_H_pow_hi[32];
extern __device__ __constant__ uint64_t d_H_pow_lo[32];

// ----------------------------------------------------------
// AES kernel entry-point declarations (visible to host and device)
// These ensure all .cu files implement the same function signatures.
// Each kernel processes a buffer of nBlocks 16-byte blocks.
// Modes: ECB (electronic codebook), CTR (counter), GCM (Galois/Counter Mode).
// ----------------------------------------------------------


// Non-templated CUDA kernels for each supported round count.  Templated
// implementations are defined privately in the .cu files and invoked from these
// wrappers so that host code can simply launch the appropriate kernel without
// relying on template instantiation across translation units.

__global__ void aes_ecb_encrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks);
__global__ void aes_ecb_encrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks);
__global__ void aes_ecb_decrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks);
__global__ void aes_ecb_decrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks);

__global__ void aes_ctr_encrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi);
__global__ void aes_ctr_encrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi);
__global__ void aes_ctr_decrypt_10(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi);
__global__ void aes_ctr_decrypt_14(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   size_t nBlocks,
                                   uint64_t ctrLo,
                                   uint64_t ctrHi);

__global__ void aes_gcm_encrypt_10(const uint8_t* __restrict__ plain,
                                   uint8_t* __restrict__ cipher,
                                   size_t nBlocks,
                                   const uint8_t* __restrict__ iv,
                                   uint8_t* __restrict__ tagOut);
__global__ void aes_gcm_encrypt_14(const uint8_t* __restrict__ plain,
                                   uint8_t* __restrict__ cipher,
                                   size_t nBlocks,
                                   const uint8_t* __restrict__ iv,
                                   uint8_t* __restrict__ tagOut);
__global__ void aes_gcm_decrypt_10(const uint8_t* __restrict__ cipher,
                                   uint8_t* __restrict__ plain,
                                   size_t nBlocks,
                                   const uint8_t* __restrict__ iv,
                                   const uint8_t* __restrict__ tag,
                                   uint8_t* __restrict__ tagOut);
__global__ void aes_gcm_decrypt_14(const uint8_t* __restrict__ cipher,
                                   uint8_t* __restrict__ plain,
                                   size_t nBlocks,
                                   const uint8_t* __restrict__ iv,
                                   const uint8_t* __restrict__ tag,
                                   uint8_t* __restrict__ tagOut);

#define aes128_ecb_encrypt aes_ecb_encrypt_10
#define aes128_ecb_decrypt aes_ecb_decrypt_10
#define aes256_ecb_encrypt aes_ecb_encrypt_14
#define aes256_ecb_decrypt aes_ecb_decrypt_14

#define aes128_ctr_encrypt aes_ctr_encrypt_10
#define aes128_ctr_decrypt aes_ctr_decrypt_10
#define aes256_ctr_encrypt aes_ctr_encrypt_14
#define aes256_ctr_decrypt aes_ctr_decrypt_14

#define aes128_gcm_encrypt aes_gcm_encrypt_10
#define aes128_gcm_decrypt aes_gcm_decrypt_10
#define aes256_gcm_encrypt aes_gcm_encrypt_14
#define aes256_gcm_decrypt aes_gcm_decrypt_14


// ----------------------------------------------------------
// Host utility functions for key expansion and constant memory setup
// (Implemented in aes_tables.cu)
// ----------------------------------------------------------
void expandKey128(const uint8_t *key16, uint32_t *roundKeys44);
void expandKey256(const uint8_t *key32, uint32_t *roundKeys60);
void init_T_tables();                     // Generate S-box, T-tables, and U-tables, copy to device const memory
void init_roundKeys(const uint32_t *rk, int nWords);  // Copy expanded round keys to device constant memory
void init_gcm_powers(const uint32_t *rk, int nRounds);

#endif  // AES_COMMON_H

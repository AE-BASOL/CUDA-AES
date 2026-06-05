#include "aes_block_device.cuh"

static __device__ __forceinline__ void aes256_ofb_crypt_block_chain(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    AesBlock feedback;
    aes_load_block(iv, feedback);
    const uint32_t *rk = d_roundKeys;

    for (size_t blk = 0; blk < nBlocks; ++blk) {
        aes_encrypt_block(feedback, rk, 14);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[blk * 16 + i] = in[blk * 16 + i] ^ feedback.b[i];
        }
    }
}

__global__ void aes256_ofb_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    aes256_ofb_crypt_block_chain(in, out, nBlocks, iv);
}

__global__ void aes256_ofb_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    aes256_ofb_crypt_block_chain(in, out, nBlocks, iv);
}

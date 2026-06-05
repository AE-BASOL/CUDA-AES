#include "aes_block_device.cuh"

__global__ void aes128_cfb_encrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint8_t feedback[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) feedback[i] = iv[i];

    const uint32_t *rk = d_roundKeys;
    for (size_t blk = 0; blk < nBlocks; ++blk) {
        AesBlock stream;
        aes_load_block(feedback, stream);
        aes_encrypt_block(stream, rk, 10);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t c = in[blk * 16 + i] ^ stream.b[i];
            out[blk * 16 + i] = c;
            feedback[i] = c;
        }
    }
}

__global__ void aes128_cfb_decrypt(const uint8_t *in, uint8_t *out, size_t nBlocks, const uint8_t *iv) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    const uint32_t *rk = d_roundKeys;

    for (size_t blk = tid; blk < nBlocks; blk += stride) {
        AesBlock stream;
        const uint8_t *feedback = (blk == 0) ? iv : in + (blk - 1) * 16;
        aes_load_block(feedback, stream);
        aes_encrypt_block(stream, rk, 10);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[blk * 16 + i] = in[blk * 16 + i] ^ stream.b[i];
        }
    }
}

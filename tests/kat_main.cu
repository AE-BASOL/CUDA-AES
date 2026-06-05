#include "aes_common.h"

#include <cuda_runtime.h>
#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;

#define CHECK_CUDA(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

std::vector<uint8_t> hex_to_bytes(const char *hex) {
    std::vector<uint8_t> out;
    for (size_t i = 0; hex[i] != '\0'; i += 2) {
        unsigned int byte = 0;
        std::sscanf(hex + i, "%2x", &byte);
        out.push_back(static_cast<uint8_t>(byte));
    }
    return out;
}

bool expect_equal(const char *name, const std::vector<uint8_t> &actual, const std::vector<uint8_t> &expected) {
    if (actual == expected) {
        std::printf("KAT PASS %s\n", name);
        return true;
    }

    std::fprintf(stderr, "KAT FAIL %s: size actual=%zu expected=%zu\n", name, actual.size(), expected.size());
    const size_t n = actual.size() < expected.size() ? actual.size() : expected.size();
    for (size_t i = 0; i < n; ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "  first mismatch byte %zu: actual=%02x expected=%02x\n", i, actual[i], expected[i]);
            break;
        }
    }
    return false;
}

void load_key(const std::vector<uint8_t> &key) {
    if (key.size() == 16) {
        std::array<uint32_t, 44> rk{};
        expandKey128(key.data(), rk.data());
        init_roundKeys(rk.data(), static_cast<int>(rk.size()));
    } else if (key.size() == 32) {
        std::array<uint32_t, 60> rk{};
        expandKey256(key.data(), rk.data());
        init_roundKeys(rk.data(), static_cast<int>(rk.size()));
    } else {
        std::fprintf(stderr, "Unsupported key size: %zu\n", key.size());
        std::exit(EXIT_FAILURE);
    }
}

void pack_counter_block_le_words(const std::vector<uint8_t> &counter, uint64_t &lo, uint64_t &hi) {
    if (counter.size() != 16) {
        std::fprintf(stderr, "CTR counter block must be 16 bytes\n");
        std::exit(EXIT_FAILURE);
    }
    std::memcpy(&lo, counter.data(), 8);
    std::memcpy(&hi, counter.data() + 8, 8);
}

std::vector<uint8_t> openssl_gcm_encrypt(const std::vector<uint8_t> &key,
                                         const std::vector<uint8_t> &iv,
                                         const std::vector<uint8_t> &plain,
                                         std::vector<uint8_t> *tag) {
    const EVP_CIPHER *cipher = key.size() == 16 ? EVP_aes_128_gcm() : EVP_aes_256_gcm();
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) std::exit(EXIT_FAILURE);

    std::vector<uint8_t> out(plain.size());
    int out_len = 0;
    int total = 0;
    EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr);
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(iv.size()), nullptr);
    EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), iv.data());
    EVP_EncryptUpdate(ctx, out.data(), &out_len, plain.data(), static_cast<int>(plain.size()));
    total += out_len;
    EVP_EncryptFinal_ex(ctx, out.data() + total, &out_len);
    total += out_len;
    out.resize(static_cast<size_t>(total));
    tag->assign(16, 0);
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag->data());
    EVP_CIPHER_CTX_free(ctx);
    return out;
}

std::vector<uint8_t> run_ecb(const std::vector<uint8_t> &key,
                             const std::vector<uint8_t> &input,
                             bool decrypt) {
    load_key(key);
    const size_t n_blocks = input.size() / 16;
    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, input.size()));
    CHECK_CUDA(cudaMalloc(&d_out, input.size()));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));

    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n_blocks + block.x - 1) / block.x));
    if (key.size() == 16 && !decrypt) aes128_ecb_encrypt<<<grid, block>>>(d_in, d_out, n_blocks);
    else if (key.size() == 16) aes128_ecb_decrypt<<<grid, block>>>(d_in, d_out, n_blocks);
    else if (!decrypt) aes256_ecb_encrypt<<<grid, block>>>(d_in, d_out, n_blocks);
    else aes256_ecb_decrypt<<<grid, block>>>(d_in, d_out, n_blocks);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> out(input.size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return out;
}

std::vector<uint8_t> run_ctr(const std::vector<uint8_t> &key,
                             const std::vector<uint8_t> &counter,
                             const std::vector<uint8_t> &input) {
    load_key(key);
    const size_t n_blocks = input.size() / 16;
    uint64_t lo = 0;
    uint64_t hi = 0;
    pack_counter_block_le_words(counter, lo, hi);

    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, input.size()));
    CHECK_CUDA(cudaMalloc(&d_out, input.size()));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n_blocks + block.x - 1) / block.x));
    if (key.size() == 16) aes128_ctr_encrypt<<<grid, block>>>(d_in, d_out, n_blocks, lo, hi);
    else aes256_ctr_encrypt<<<grid, block>>>(d_in, d_out, n_blocks, lo, hi);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> out(input.size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return out;
}

std::vector<uint8_t> run_cbc(const std::vector<uint8_t> &key,
                             const std::vector<uint8_t> &iv,
                             const std::vector<uint8_t> &input,
                             bool decrypt) {
    if (iv.size() != 16) {
        std::fprintf(stderr, "CBC IV must be 16 bytes\n");
        std::exit(EXIT_FAILURE);
    }
    load_key(key);
    const size_t n_blocks = input.size() / 16;

    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    uint8_t *d_iv = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, input.size()));
    CHECK_CUDA(cudaMalloc(&d_out, input.size()));
    CHECK_CUDA(cudaMalloc(&d_iv, iv.size()));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n_blocks + block.x - 1) / block.x));
    if (key.size() == 16 && !decrypt) aes128_cbc_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else if (key.size() == 16) aes128_cbc_decrypt<<<grid, block>>>(d_in, d_out, n_blocks, d_iv);
    else if (!decrypt) aes256_cbc_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else aes256_cbc_decrypt<<<grid, block>>>(d_in, d_out, n_blocks, d_iv);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> out(input.size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_iv));
    return out;
}

std::vector<uint8_t> run_cfb(const std::vector<uint8_t> &key,
                             const std::vector<uint8_t> &iv,
                             const std::vector<uint8_t> &input,
                             bool decrypt) {
    if (iv.size() != 16) {
        std::fprintf(stderr, "CFB-128 IV must be 16 bytes\n");
        std::exit(EXIT_FAILURE);
    }
    load_key(key);
    const size_t n_blocks = input.size() / 16;

    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    uint8_t *d_iv = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, input.size()));
    CHECK_CUDA(cudaMalloc(&d_out, input.size()));
    CHECK_CUDA(cudaMalloc(&d_iv, iv.size()));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
    dim3 block(kThreads);
    dim3 grid(static_cast<unsigned>((n_blocks + block.x - 1) / block.x));
    if (key.size() == 16 && !decrypt) aes128_cfb_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else if (key.size() == 16) aes128_cfb_decrypt<<<grid, block>>>(d_in, d_out, n_blocks, d_iv);
    else if (!decrypt) aes256_cfb_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else aes256_cfb_decrypt<<<grid, block>>>(d_in, d_out, n_blocks, d_iv);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> out(input.size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_iv));
    return out;
}

std::vector<uint8_t> run_ofb(const std::vector<uint8_t> &key,
                             const std::vector<uint8_t> &iv,
                             const std::vector<uint8_t> &input,
                             bool decrypt) {
    if (iv.size() != 16) {
        std::fprintf(stderr, "OFB IV must be 16 bytes\n");
        std::exit(EXIT_FAILURE);
    }
    load_key(key);
    const size_t n_blocks = input.size() / 16;

    uint8_t *d_in = nullptr;
    uint8_t *d_out = nullptr;
    uint8_t *d_iv = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, input.size()));
    CHECK_CUDA(cudaMalloc(&d_out, input.size()));
    CHECK_CUDA(cudaMalloc(&d_iv, iv.size()));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
    if (key.size() == 16 && !decrypt) aes128_ofb_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else if (key.size() == 16) aes128_ofb_decrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else if (!decrypt) aes256_ofb_encrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    else aes256_ofb_decrypt<<<1, 1>>>(d_in, d_out, n_blocks, d_iv);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> out(input.size());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_iv));
    return out;
}

std::vector<uint8_t> run_gcm_encrypt(const std::vector<uint8_t> &key,
                                     const std::vector<uint8_t> &iv,
                                     const std::vector<uint8_t> &plain,
                                     std::vector<uint8_t> *tag) {
    load_key(key);
    const size_t n_blocks = plain.size() / 16;
    uint8_t *d_plain = nullptr;
    uint8_t *d_cipher = nullptr;
    uint8_t *d_iv = nullptr;
    uint8_t *d_tag = nullptr;
    CHECK_CUDA(cudaMalloc(&d_plain, plain.size()));
    CHECK_CUDA(cudaMalloc(&d_cipher, plain.size()));
    CHECK_CUDA(cudaMalloc(&d_iv, iv.size()));
    CHECK_CUDA(cudaMalloc(&d_tag, 16));
    CHECK_CUDA(cudaMemcpy(d_plain, plain.data(), plain.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
    if (key.size() == 16) aes128_gcm_encrypt<<<1, kThreads>>>(d_plain, d_cipher, n_blocks, d_iv, d_tag);
    else aes256_gcm_encrypt<<<1, kThreads>>>(d_plain, d_cipher, n_blocks, d_iv, d_tag);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> cipher(plain.size());
    tag->assign(16, 0);
    CHECK_CUDA(cudaMemcpy(cipher.data(), d_cipher, cipher.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tag->data(), d_tag, tag->size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_plain));
    CHECK_CUDA(cudaFree(d_cipher));
    CHECK_CUDA(cudaFree(d_iv));
    CHECK_CUDA(cudaFree(d_tag));
    return cipher;
}

std::vector<uint8_t> run_gcm_decrypt(const std::vector<uint8_t> &key,
                                     const std::vector<uint8_t> &iv,
                                     const std::vector<uint8_t> &cipher,
                                     const std::vector<uint8_t> &expected_tag,
                                     std::vector<uint8_t> *computed_tag) {
    load_key(key);
    const size_t n_blocks = cipher.size() / 16;
    uint8_t *d_cipher = nullptr;
    uint8_t *d_plain = nullptr;
    uint8_t *d_iv = nullptr;
    uint8_t *d_tag_in = nullptr;
    uint8_t *d_tag_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cipher, cipher.size()));
    CHECK_CUDA(cudaMalloc(&d_plain, cipher.size()));
    CHECK_CUDA(cudaMalloc(&d_iv, iv.size()));
    CHECK_CUDA(cudaMalloc(&d_tag_in, 16));
    CHECK_CUDA(cudaMalloc(&d_tag_out, 16));
    CHECK_CUDA(cudaMemcpy(d_cipher, cipher.data(), cipher.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv, iv.data(), iv.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tag_in, expected_tag.data(), expected_tag.size(), cudaMemcpyHostToDevice));
    if (key.size() == 16) aes128_gcm_decrypt<<<1, kThreads>>>(d_cipher, d_plain, n_blocks, d_iv, d_tag_in, d_tag_out);
    else aes256_gcm_decrypt<<<1, kThreads>>>(d_cipher, d_plain, n_blocks, d_iv, d_tag_in, d_tag_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> plain(cipher.size());
    computed_tag->assign(16, 0);
    CHECK_CUDA(cudaMemcpy(plain.data(), d_plain, plain.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(computed_tag->data(), d_tag_out, computed_tag->size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_cipher));
    CHECK_CUDA(cudaFree(d_plain));
    CHECK_CUDA(cudaFree(d_iv));
    CHECK_CUDA(cudaFree(d_tag_in));
    CHECK_CUDA(cudaFree(d_tag_out));
    return plain;
}

std::vector<uint8_t> run_ccm_encrypt(const std::vector<uint8_t> &key,
                                     const std::vector<uint8_t> &nonce,
                                     const std::vector<uint8_t> &plain,
                                     std::vector<uint8_t> *tag) {
    if (nonce.size() != 12) {
        std::fprintf(stderr, "CCM nonce must be 12 bytes for the benchmark scope\n");
        std::exit(EXIT_FAILURE);
    }
    load_key(key);
    const size_t n_blocks = plain.size() / 16;
    uint8_t *d_plain = nullptr;
    uint8_t *d_cipher = nullptr;
    uint8_t *d_nonce = nullptr;
    uint8_t *d_tag = nullptr;
    CHECK_CUDA(cudaMalloc(&d_plain, plain.size()));
    CHECK_CUDA(cudaMalloc(&d_cipher, plain.size()));
    CHECK_CUDA(cudaMalloc(&d_nonce, nonce.size()));
    CHECK_CUDA(cudaMalloc(&d_tag, 16));
    CHECK_CUDA(cudaMemcpy(d_plain, plain.data(), plain.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice));
    if (key.size() == 16) aes128_ccm_encrypt<<<1, kThreads>>>(d_plain, d_cipher, n_blocks, d_nonce, d_tag);
    else aes256_ccm_encrypt<<<1, kThreads>>>(d_plain, d_cipher, n_blocks, d_nonce, d_tag);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> cipher(plain.size());
    tag->assign(16, 0);
    CHECK_CUDA(cudaMemcpy(cipher.data(), d_cipher, cipher.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(tag->data(), d_tag, tag->size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_plain));
    CHECK_CUDA(cudaFree(d_cipher));
    CHECK_CUDA(cudaFree(d_nonce));
    CHECK_CUDA(cudaFree(d_tag));
    return cipher;
}

std::vector<uint8_t> run_ccm_decrypt(const std::vector<uint8_t> &key,
                                     const std::vector<uint8_t> &nonce,
                                     const std::vector<uint8_t> &cipher,
                                     const std::vector<uint8_t> &expected_tag,
                                     std::vector<uint8_t> *computed_tag) {
    if (nonce.size() != 12) {
        std::fprintf(stderr, "CCM nonce must be 12 bytes for the benchmark scope\n");
        std::exit(EXIT_FAILURE);
    }
    load_key(key);
    const size_t n_blocks = cipher.size() / 16;
    uint8_t *d_cipher = nullptr;
    uint8_t *d_plain = nullptr;
    uint8_t *d_nonce = nullptr;
    uint8_t *d_tag_in = nullptr;
    uint8_t *d_tag_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_cipher, cipher.size()));
    CHECK_CUDA(cudaMalloc(&d_plain, cipher.size()));
    CHECK_CUDA(cudaMalloc(&d_nonce, nonce.size()));
    CHECK_CUDA(cudaMalloc(&d_tag_in, 16));
    CHECK_CUDA(cudaMalloc(&d_tag_out, 16));
    CHECK_CUDA(cudaMemcpy(d_cipher, cipher.data(), cipher.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_nonce, nonce.data(), nonce.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_tag_in, expected_tag.data(), expected_tag.size(), cudaMemcpyHostToDevice));
    if (key.size() == 16) aes128_ccm_decrypt<<<1, kThreads>>>(d_cipher, d_plain, n_blocks, d_nonce, d_tag_in, d_tag_out);
    else aes256_ccm_decrypt<<<1, kThreads>>>(d_cipher, d_plain, n_blocks, d_nonce, d_tag_in, d_tag_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> plain(cipher.size());
    computed_tag->assign(16, 0);
    CHECK_CUDA(cudaMemcpy(plain.data(), d_plain, plain.size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(computed_tag->data(), d_tag_out, computed_tag->size(), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_cipher));
    CHECK_CUDA(cudaFree(d_plain));
    CHECK_CUDA(cudaFree(d_nonce));
    CHECK_CUDA(cudaFree(d_tag_in));
    CHECK_CUDA(cudaFree(d_tag_out));
    return plain;
}

bool run_all() {
    bool ok = true;
    init_T_tables();

    const auto ecb128_key = hex_to_bytes("2b7e151628aed2a6abf7158809cf4f3c");
    const auto ecb256_key = hex_to_bytes("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4");
    const auto one_block_plain = hex_to_bytes("6bc1bee22e409f96e93d7e117393172a");
    const auto ecb128_cipher = hex_to_bytes("3ad77bb40d7a3660a89ecaf32466ef97");
    const auto ecb256_cipher = hex_to_bytes("f3eed1bdb5d2a03c064b5a7e3db181f8");
    ok &= expect_equal("ECB-128 encrypt", run_ecb(ecb128_key, one_block_plain, false), ecb128_cipher);
    ok &= expect_equal("ECB-128 decrypt", run_ecb(ecb128_key, ecb128_cipher, true), one_block_plain);
    ok &= expect_equal("ECB-256 encrypt", run_ecb(ecb256_key, one_block_plain, false), ecb256_cipher);
    ok &= expect_equal("ECB-256 decrypt", run_ecb(ecb256_key, ecb256_cipher, true), one_block_plain);

    const auto ctr_counter = hex_to_bytes("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
    const auto ctr_plain = hex_to_bytes(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51"
        "30c81c46a35ce411e5fbc1191a0a52ef"
        "f69f2445df4f9b17ad2b417be66c3710");
    const auto ctr128_cipher = hex_to_bytes(
        "874d6191b620e3261bef6864990db6ce"
        "9806f66b7970fdff8617187bb9fffdff"
        "5ae4df3edbd5d35e5b4f09020db03eab"
        "1e031dda2fbe03d1792170a0f3009cee");
    const auto ctr256_cipher = hex_to_bytes(
        "601ec313775789a5b7a7f504bbf3d228"
        "f443e3ca4d62b59aca84e990cacaf5c5"
        "2b0930daa23de94ce87017ba2d84988d"
        "dfc9c58db67aada613c2dd08457941a6");
    ok &= expect_equal("CTR-128 encrypt", run_ctr(ecb128_key, ctr_counter, ctr_plain), ctr128_cipher);
    ok &= expect_equal("CTR-128 decrypt", run_ctr(ecb128_key, ctr_counter, ctr128_cipher), ctr_plain);
    ok &= expect_equal("CTR-256 encrypt", run_ctr(ecb256_key, ctr_counter, ctr_plain), ctr256_cipher);
    ok &= expect_equal("CTR-256 decrypt", run_ctr(ecb256_key, ctr_counter, ctr256_cipher), ctr_plain);

    const auto cbc_iv = hex_to_bytes("000102030405060708090a0b0c0d0e0f");
    const auto cbc_plain = ctr_plain;
    const auto cbc128_cipher = hex_to_bytes(
        "7649abac8119b246cee98e9b12e9197d"
        "5086cb9b507219ee95db113a917678b2"
        "73bed6b8e3c1743b7116e69e22229516"
        "3ff1caa1681fac09120eca307586e1a7");
    const auto cbc256_cipher = hex_to_bytes(
        "f58c4c04d6e5f1ba779eabfb5f7bfbd6"
        "9cfc4e967edb808d679f777bc6702c7d"
        "39f23369a9d9bacfa530e26304231461"
        "b2eb05e2c39be9fcda6c19078c6a9d1b");
    ok &= expect_equal("CBC-128 encrypt", run_cbc(ecb128_key, cbc_iv, cbc_plain, false), cbc128_cipher);
    ok &= expect_equal("CBC-128 decrypt", run_cbc(ecb128_key, cbc_iv, cbc128_cipher, true), cbc_plain);
    ok &= expect_equal("CBC-256 encrypt", run_cbc(ecb256_key, cbc_iv, cbc_plain, false), cbc256_cipher);
    ok &= expect_equal("CBC-256 decrypt", run_cbc(ecb256_key, cbc_iv, cbc256_cipher, true), cbc_plain);

    const auto cfb128_cipher = hex_to_bytes(
        "3b3fd92eb72dad20333449f8e83cfb4a"
        "c8a64537a0b3a93fcde3cdad9f1ce58b"
        "26751f67a3cbb140b1808cf187a4f4df"
        "c04b05357c5d1c0eeac4c66f9ff7f2e6");
    const auto cfb256_cipher = hex_to_bytes(
        "dc7e84bfda79164b7ecd8486985d3860"
        "39ffed143b28b1c832113c6331e5407b"
        "df10132415e54b92a13ed0a8267ae2f9"
        "75a385741ab9cef82031623d55b1e471");
    ok &= expect_equal("CFB-128 encrypt", run_cfb(ecb128_key, cbc_iv, cbc_plain, false), cfb128_cipher);
    ok &= expect_equal("CFB-128 decrypt", run_cfb(ecb128_key, cbc_iv, cfb128_cipher, true), cbc_plain);
    ok &= expect_equal("CFB-256 encrypt", run_cfb(ecb256_key, cbc_iv, cbc_plain, false), cfb256_cipher);
    ok &= expect_equal("CFB-256 decrypt", run_cfb(ecb256_key, cbc_iv, cfb256_cipher, true), cbc_plain);

    const auto ofb128_cipher = hex_to_bytes(
        "3b3fd92eb72dad20333449f8e83cfb4a"
        "7789508d16918f03f53c52dac54ed825"
        "9740051e9c5fecf64344f7a82260edcc"
        "304c6528f659c77866a510d9c1d6ae5e");
    const auto ofb256_cipher = hex_to_bytes(
        "dc7e84bfda79164b7ecd8486985d3860"
        "4febdc6740d20b3ac88f6ad82a4fb08d"
        "71ab47a086e86eedf39d1c5bba97c408"
        "0126141d67f37be8538f5a8be740e484");
    ok &= expect_equal("OFB-128 encrypt", run_ofb(ecb128_key, cbc_iv, cbc_plain, false), ofb128_cipher);
    ok &= expect_equal("OFB-128 decrypt", run_ofb(ecb128_key, cbc_iv, ofb128_cipher, true), cbc_plain);
    ok &= expect_equal("OFB-256 encrypt", run_ofb(ecb256_key, cbc_iv, cbc_plain, false), ofb256_cipher);
    ok &= expect_equal("OFB-256 decrypt", run_ofb(ecb256_key, cbc_iv, ofb256_cipher, true), cbc_plain);

    const auto zero128_key = hex_to_bytes("00000000000000000000000000000000");
    const auto zero256_key = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000000");
    const auto zero_iv = hex_to_bytes("000000000000000000000000");
    const auto gcm_plain = hex_to_bytes("00000000000000000000000000000000");
    const auto gcm128_cipher = hex_to_bytes("0388dace60b6a392f328c2b971b2fe78");
    const auto gcm128_tag = hex_to_bytes("ab6e47d42cec13bdf53a67b21257bddf");

    std::vector<uint8_t> tag;
    std::vector<uint8_t> cipher = run_gcm_encrypt(zero128_key, zero_iv, gcm_plain, &tag);
    ok &= expect_equal("GCM-128 encrypt ciphertext", cipher, gcm128_cipher);
    ok &= expect_equal("GCM-128 encrypt tag", tag, gcm128_tag);
    std::vector<uint8_t> computed_tag;
    std::vector<uint8_t> plain = run_gcm_decrypt(zero128_key, zero_iv, gcm128_cipher, gcm128_tag, &computed_tag);
    ok &= expect_equal("GCM-128 decrypt plaintext", plain, gcm_plain);
    ok &= expect_equal("GCM-128 decrypt tag", computed_tag, gcm128_tag);

    std::vector<uint8_t> openssl_tag256;
    std::vector<uint8_t> openssl_cipher256 = openssl_gcm_encrypt(zero256_key, zero_iv, gcm_plain, &openssl_tag256);
    cipher = run_gcm_encrypt(zero256_key, zero_iv, gcm_plain, &tag);
    ok &= expect_equal("GCM-256 encrypt ciphertext", cipher, openssl_cipher256);
    ok &= expect_equal("GCM-256 encrypt tag", tag, openssl_tag256);
    plain = run_gcm_decrypt(zero256_key, zero_iv, openssl_cipher256, openssl_tag256, &computed_tag);
    ok &= expect_equal("GCM-256 decrypt plaintext", plain, gcm_plain);
    ok &= expect_equal("GCM-256 decrypt tag", computed_tag, openssl_tag256);

    std::vector<uint8_t> wrong_tag = gcm128_tag;
    wrong_tag[0] ^= 0x01;
    plain = run_gcm_decrypt(zero128_key, zero_iv, gcm128_cipher, wrong_tag, &computed_tag);
    if (computed_tag == wrong_tag) {
        std::fprintf(stderr, "KAT FAIL GCM-128 wrong tag was accepted\n");
        ok = false;
    } else {
        std::printf("KAT PASS GCM-128 wrong tag rejected\n");
    }

    std::vector<uint8_t> tampered_cipher = gcm128_cipher;
    tampered_cipher[0] ^= 0x01;
    plain = run_gcm_decrypt(zero128_key, zero_iv, tampered_cipher, gcm128_tag, &computed_tag);
    if (computed_tag == gcm128_tag) {
        std::fprintf(stderr, "KAT FAIL GCM-128 tampered ciphertext was accepted\n");
        ok = false;
    } else {
        std::printf("KAT PASS GCM-128 tampered ciphertext rejected\n");
    }

    const auto ccm_nonce = hex_to_bytes("101112131415161718191a1b");
    const auto ccm_plain = hex_to_bytes(
        "6bc1bee22e409f96e93d7e117393172a"
        "ae2d8a571e03ac9c9eb76fac45af8e51");
    const auto ccm128_cipher = hex_to_bytes(
        "76c0f267fbe2820aad1470f1fb0340b0"
        "d231bdebb290f27387ea727570ae567d");
    const auto ccm128_tag = hex_to_bytes("ad5ea85fe260bfa769cea1bff028af7f");
    const auto ccm256_cipher = hex_to_bytes(
        "b3c2479fd407e32f7f2482e0c9dc89dd"
        "70d77c6daa191fd1a1e8a0eb8020e1b2");
    const auto ccm256_tag = hex_to_bytes("dd7f692954ea5b452323334990655935");

    cipher = run_ccm_encrypt(ecb128_key, ccm_nonce, ccm_plain, &tag);
    ok &= expect_equal("CCM-128 encrypt ciphertext", cipher, ccm128_cipher);
    ok &= expect_equal("CCM-128 encrypt tag", tag, ccm128_tag);
    plain = run_ccm_decrypt(ecb128_key, ccm_nonce, ccm128_cipher, ccm128_tag, &computed_tag);
    ok &= expect_equal("CCM-128 decrypt plaintext", plain, ccm_plain);
    ok &= expect_equal("CCM-128 decrypt tag", computed_tag, ccm128_tag);

    cipher = run_ccm_encrypt(ecb256_key, ccm_nonce, ccm_plain, &tag);
    ok &= expect_equal("CCM-256 encrypt ciphertext", cipher, ccm256_cipher);
    ok &= expect_equal("CCM-256 encrypt tag", tag, ccm256_tag);
    plain = run_ccm_decrypt(ecb256_key, ccm_nonce, ccm256_cipher, ccm256_tag, &computed_tag);
    ok &= expect_equal("CCM-256 decrypt plaintext", plain, ccm_plain);
    ok &= expect_equal("CCM-256 decrypt tag", computed_tag, ccm256_tag);

    std::vector<uint8_t> wrong_ccm_tag = ccm128_tag;
    wrong_ccm_tag[0] ^= 0x01;
    plain = run_ccm_decrypt(ecb128_key, ccm_nonce, ccm128_cipher, wrong_ccm_tag, &computed_tag);
    if (computed_tag == wrong_ccm_tag) {
        std::fprintf(stderr, "KAT FAIL CCM-128 wrong tag was accepted\n");
        ok = false;
    } else {
        std::printf("KAT PASS CCM-128 wrong tag rejected\n");
    }

    return ok;
}

}  // namespace

int main() {
    return run_all() ? EXIT_SUCCESS : EXIT_FAILURE;
}

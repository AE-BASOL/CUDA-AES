#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <getopt.h>
#include <immintrin.h>
#include <openssl/evp.h>
#include "aes_common.h"
#include "profiling_helpers.h"

// -------------------------------
// Error handling macro
// -------------------------------
#define CHECK_CUDA(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// -------------------------------
// Constants and parameters
// -------------------------------
constexpr int THREADS_PER_BLOCK = 256;
constexpr int NUM_RUNS          = 5;
static const size_t SIZES[]     = {1ull<<20, 10ull<<20, 100ull<<20, 1ull<<30};
static const char*  MODES[]     = {"ecb-128","ecb-256","ctr-128","ctr-256","gcm-128","gcm-256"};

// -------------------------------
// CTR helper
// -------------------------------
static void packCtr(const uint8_t iv[12], uint64_t &lo, uint64_t &hi) {
    uint32_t w0=0,w1=0,w2=0; memcpy(&w0,iv,4); memcpy(&w1,iv+4,4); memcpy(&w2,iv+8,4);
    uint32_t w3=0x01000000u; lo = (uint64_t)w0 | ((uint64_t)w1<<32); hi = (uint64_t)w2 | ((uint64_t)w3<<32);
}

// -------------------------------
// Device GF multiply used for --gf-mult and GCM debug
// -------------------------------
__device__ inline void gf_mul128_dev(uint64_t &Ah, uint64_t &Al, uint64_t Bh, uint64_t Bl) {
    uint64_t Zh=0, Zl=0, Vh=Bh, Vl=Bl; const uint64_t R=0xE100000000000000ULL;
    for(int i=0;i<128;++i){
        if(Al & 1ULL){ Zl^=Vl; Zh^=Vh; }
        bool carry = Vl & 1ULL;
        Vl = (Vl>>1) | (Vh<<63); Vh >>=1; if(carry) Vh^=R;
        Al = (Al>>1) | (Ah<<63); Ah >>=1;
    }
    Ah=Zh; Al=Zl;
}

// Kernel performing many GF multiplies per thread
__global__ void gf_mult_kernel(uint64_t *out) {
    uint64_t Ah=0x0123456789abcdefULL, Al=0xfedcba9876543210ULL;
    uint64_t Bh=0x0fedcba987654321ULL, Bl=0x1234567890abcdefULL;
    for(int i=0;i<1000000;i++) {
        gf_mul128_dev(Ah,Al,Bh,Bl);
        Bh += 1; Bl += 1;
    }
    out[threadIdx.x] = Ah ^ Al ^ Bh ^ Bl;
}

// Kernel computing per-thread partial GHASH
__global__ void gcm_partial_kernel(const uint8_t *cipher, size_t nBlocks,
                                   uint64_t Hh, uint64_t Hl,
                                   uint64_t *outH, uint64_t *outL) {
    int tid = threadIdx.x;
    size_t start = tid * nBlocks / blockDim.x;
    size_t end   = (tid+1) * nBlocks / blockDim.x;
    uint64_t Xh=0, Xl=0;
    for(size_t i=start;i<end;++i){
        uint64_t cl=((const uint64_t*)cipher)[2*i];
        uint64_t ch=((const uint64_t*)cipher)[2*i+1];
        Xl ^= cl; Xh ^= ch; gf_mul128_dev(Xh,Xl,Hh,Hl);
    }
    outH[tid]=Xh; outL[tid]=Xl;
}

// -------------------------------
// OpenSSL throughput helper
// -------------------------------
static double cpu_aes_throughput(const void* src, size_t bytes,
                                 const unsigned char* key, int bits,
                                 bool decrypt, const EVP_CIPHER* (*cipherSel)()) {
    std::vector<unsigned char> buf(bytes);
    std::vector<unsigned char> iv(16,0);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    const EVP_CIPHER *cipher = cipherSel();
    if(decrypt) EVP_DecryptInit_ex(ctx, cipher, nullptr, key, iv.data());
    else        EVP_EncryptInit_ex(ctx, cipher, nullptr, key, iv.data());
    EVP_CIPHER_CTX_set_padding(ctx,0);
    auto t0=std::chrono::high_resolution_clock::now();
    int outLen=0,total=0;
    if(decrypt) EVP_DecryptUpdate(ctx, buf.data(), &outLen, (const unsigned char*)src, (int)bytes);
    else        EVP_EncryptUpdate(ctx, buf.data(), &outLen, (const unsigned char*)src, (int)bytes);
    total += outLen;
    if(decrypt) EVP_DecryptFinal_ex(ctx, buf.data()+total, &outLen);
    else        EVP_EncryptFinal_ex(ctx, buf.data()+total, &outLen);
    total += outLen;
    auto t1=std::chrono::high_resolution_clock::now();
    EVP_CIPHER_CTX_free(ctx);
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double gib=(double)bytes/(double)(1ull<<30);
    return gib/(ms/1000.0);
}

// -------------------------------
// Helper to generate random bytes
// -------------------------------
static void fill_random(uint8_t *buf, size_t n, std::mt19937_64 &rng) {
    for(size_t i=0;i<n;++i) buf[i] = static_cast<uint8_t>(rng() & 0xFF);
}

// -------------------------------
// CTR preview routine
// -------------------------------
static int ctr_preview() {
    std::mt19937_64 rng(42);
    std::vector<uint8_t> key(16); fill_random(key.data(),16,rng);
    std::vector<uint8_t> iv(12);  fill_random(iv.data(),12,rng);

    std::vector<uint32_t> rk(44); expandKey128(key.data(), rk.data());
    init_roundKeys(rk.data(), (int)rk.size());

    uint8_t *d_in,*d_out; CHECK_CUDA(cudaMalloc(&d_in,32)); CHECK_CUDA(cudaMalloc(&d_out,32));
    CHECK_CUDA(cudaMemset(d_in,0,32));
    uint64_t lo=0,hi=0; packCtr(iv.data(),lo,hi);
    NVTX_PUSH("CTR_PREVIEW");
    aes128_ctr_encrypt<<<1,1>>>(d_in,d_out,2,lo,hi);
    CHECK_CUDA(cudaDeviceSynchronize());
    NVTX_POP();
    uint8_t h_out[32]; CHECK_CUDA(cudaMemcpy(h_out,d_out,32,cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));

    printf("CTR_PREVIEW,");
    for(int i=0;i<32;i++){ printf("%02x", h_out[i]); if(i==15) printf(","); }
    printf("\n");
    return 0;
}

// -------------------------------
// GF multiply benchmark
// -------------------------------
static int gf_mult_bench() {
    std::filesystem::create_directories("bench");
    // CPU part
    double ms_cpu=0.0; {
        __m128i a = _mm_set_epi64x(0x0123456789abcdefULL,0xfedcba9876543210ULL);
        __m128i b = _mm_set_epi64x(0x0fedcba987654321ULL,0x1234567890abcdefULL);
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<1000000;i++) {
            __m128i r = _mm_clmulepi64_si128(a,b,0x00);
            a = _mm_xor_si128(a,r);
            b = _mm_xor_si128(b,r);
        }
        auto t1=std::chrono::high_resolution_clock::now();
        ms_cpu=std::chrono::duration<double,std::milli>(t1-t0).count();
    }
    double gbps_cpu = (1000000.0*128/1e9) / (ms_cpu/1000.0);

    // GPU part
    double ms_gpu=0.0; double gbps_gpu=0.0; {
        uint64_t *d_out; CHECK_CUDA(cudaMalloc(&d_out, THREADS_PER_BLOCK*sizeof(uint64_t)));
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        gf_mult_kernel<<<1,THREADS_PER_BLOCK>>>(d_out);
        cudaEventRecord(e); CHECK_CUDA(cudaEventSynchronize(e));
        cudaEventElapsedTime(&ms_gpu,s,e); CHECK_CUDA(cudaFree(d_out));
        gbps_gpu = (1000000.0*THREADS_PER_BLOCK*128/1e9) / (ms_gpu/1000.0);
    }

    std::ofstream f("bench/gf_mult.csv", std::ios::app);
    f << "SRC,CPU,1000000," << ms_cpu << ',' << gbps_cpu << "\n";
    f << "SRC,GPU," << (1000000*THREADS_PER_BLOCK) << ',' << ms_gpu << ',' << gbps_gpu << "\n";
    std::cout << "GF_MULT CPU "<<gbps_cpu<<" Gbps\n";
    std::cout << "GF_MULT GPU "<<gbps_gpu<<" Gbps\n";
    return 0;
}

// -------------------------------
// GCM debug routine: encrypt 64B and dump partial GHASH
// -------------------------------
static int gcm_debug_run() {
    std::filesystem::create_directories("bench");
    std::mt19937_64 rng(123);
    const size_t bytes=64; size_t nBlocks=bytes/16;
    uint8_t *h_plain,*h_cipher; CHECK_CUDA(cudaMallocHost(&h_plain,bytes)); CHECK_CUDA(cudaMallocHost(&h_cipher,bytes));
    fill_random(h_plain,bytes,rng);
    std::vector<uint8_t> key(16); fill_random(key.data(),16,rng);
    std::vector<uint8_t> iv(12);  fill_random(iv.data(),12,rng);
    std::vector<uint32_t> rk(44); expandKey128(key.data(), rk.data());
    init_roundKeys(rk.data(), (int)rk.size());
    uint8_t *d_plain,*d_cipher,*d_tag,*d_iv; CHECK_CUDA(cudaMalloc(&d_plain,bytes)); CHECK_CUDA(cudaMalloc(&d_cipher,bytes)); CHECK_CUDA(cudaMalloc(&d_tag,16)); CHECK_CUDA(cudaMalloc(&d_iv,12));
    CHECK_CUDA(cudaMemcpy(d_plain,h_plain,bytes,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iv,iv.data(),12,cudaMemcpyHostToDevice));
    aes128_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_plain,d_cipher,nBlocks,d_iv,d_tag);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_cipher,d_cipher,bytes,cudaMemcpyDeviceToHost));

    // compute H = AES_k(0)
    uint8_t *d_zero,*d_h; CHECK_CUDA(cudaMalloc(&d_zero,16)); CHECK_CUDA(cudaMalloc(&d_h,16));
    CHECK_CUDA(cudaMemset(d_zero,0,16));
    aes128_ecb_encrypt<<<1,1>>>(d_zero,d_h,1);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint8_t hbuf[16]; CHECK_CUDA(cudaMemcpy(hbuf,d_h,16,cudaMemcpyDeviceToHost));
    uint64_t Hl=((uint64_t*)hbuf)[0]; uint64_t Hh=((uint64_t*)hbuf)[1];
    CHECK_CUDA(cudaFree(d_zero)); CHECK_CUDA(cudaFree(d_h));

    // partial GHASH
    uint64_t *d_ph,*d_pl; CHECK_CUDA(cudaMalloc(&d_ph,THREADS_PER_BLOCK*sizeof(uint64_t))); CHECK_CUDA(cudaMalloc(&d_pl,THREADS_PER_BLOCK*sizeof(uint64_t)));
    gcm_partial_kernel<<<1,THREADS_PER_BLOCK>>>(d_cipher,nBlocks,Hh,Hl,d_ph,d_pl);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<uint64_t> ph(THREADS_PER_BLOCK), pl(THREADS_PER_BLOCK);
    CHECK_CUDA(cudaMemcpy(ph.data(),d_ph,THREADS_PER_BLOCK*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(pl.data(),d_pl,THREADS_PER_BLOCK*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_ph)); CHECK_CUDA(cudaFree(d_pl));
    std::ofstream out("bench/ghash_partials.txt");
    for(int i=0;i<THREADS_PER_BLOCK;i++)
        out << i << "," << std::hex << ph[i] << "," << pl[i] << std::dec << "\n";

    CHECK_CUDA(cudaFree(d_plain)); CHECK_CUDA(cudaFree(d_cipher)); CHECK_CUDA(cudaFree(d_tag)); CHECK_CUDA(cudaFree(d_iv));
    CHECK_CUDA(cudaFreeHost(h_plain)); CHECK_CUDA(cudaFreeHost(h_cipher));
    std::cout << "GHASH partials written to bench/ghash_partials.txt\n";
    return 0;
}

// -------------------------------
// Main benchmark loop
// -------------------------------
int main(int argc, char** argv) {
    int blockOverride = THREADS_PER_BLOCK;
    bool decrypt=false, doCtrPreview=false, doGcmDebug=false, doGfMult=false;

    enum { OPT_CTR_PREVIEW=1000, OPT_GCM_DEBUG, OPT_GF_MULT };
    static struct option opts[] = {
        {"block", required_argument, nullptr, 'b'},
        {"decrypt", no_argument, nullptr, 'd'},
        {"ctr-preview", no_argument, nullptr, OPT_CTR_PREVIEW},
        {"gcm-debug", no_argument, nullptr, OPT_GCM_DEBUG},
        {"gf-mult", no_argument, nullptr, OPT_GF_MULT},
        {"help", no_argument, nullptr, 'h'},
        {0,0,0,0}
    };
    while(true){
        int idx=0; int c=getopt_long(argc,argv,"b:dh",opts,&idx); if(c==-1) break;
        switch(c){
            case 'b': blockOverride=atoi(optarg); break;
            case 'd': decrypt=true; break;
            case OPT_CTR_PREVIEW: doCtrPreview=true; break;
            case OPT_GCM_DEBUG: doGcmDebug=true; break;
            case OPT_GF_MULT: doGfMult=true; break;
            case 'h':
            default:
                std::cout << "Usage: "<<argv[0]<<" [--block N] [--decrypt] [--ctr-preview] [--gcm-debug] [--gf-mult]\n";
                return 0;
        }
    }

    std::filesystem::create_directories("bench");
    init_T_tables();

    if(doCtrPreview) return ctr_preview();
    if(doGcmDebug)   return gcm_debug_run();
    if(doGfMult)     return gf_mult_bench();

    std::mt19937_64 rng(12345);
    for(const char* modeStr : MODES){
        std::string mode(modeStr);
        bool isEcb = mode.find("ecb")==0;
        bool isCtr = mode.find("ctr")==0;
        bool isGcm = mode.find("gcm")==0;
        int bits = mode.find("256")!=std::string::npos ? 256 : 128;
        size_t keyBytes = bits/8;
        std::vector<uint8_t> key(keyBytes); fill_random(key.data(),keyBytes,rng);
        std::vector<uint32_t> rk(bits==128?44:60);
        if(bits==128) expandKey128(key.data(),rk.data()); else expandKey256(key.data(),rk.data());
        init_roundKeys(rk.data(), (int)rk.size());
        std::vector<uint8_t> iv(12); if(!isEcb) fill_random(iv.data(),12,rng);

        for(size_t sz : SIZES){
            size_t nBlocks=(sz+15)/16; size_t bytes=nBlocks*16;
            for(int run=1; run<=NUM_RUNS; ++run){
                uint8_t *h_in,*h_out; CHECK_CUDA(cudaMallocHost(&h_in,bytes)); CHECK_CUDA(cudaMallocHost(&h_out,bytes));
                fill_random(h_in,bytes,rng);
                uint8_t *d_in,*d_out,*d_tag=nullptr,*d_iv=nullptr;
                CHECK_CUDA(cudaMalloc(&d_in,bytes)); CHECK_CUDA(cudaMalloc(&d_out,bytes));
                if(isGcm) { CHECK_CUDA(cudaMalloc(&d_tag,16)); CHECK_CUDA(cudaMalloc(&d_iv,12)); CHECK_CUDA(cudaMemcpy(d_iv,iv.data(),12,cudaMemcpyHostToDevice)); }
                CHECK_CUDA(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));
                dim3 block(blockOverride); dim3 grid((unsigned)((nBlocks+block.x-1)/block.x));
                cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
                cudaEventRecord(s);
                if(!decrypt){
                    if(isEcb && bits==128){ NVTX_PUSH("ecb128_enc"); aes128_ecb_encrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isEcb && bits==256){ NVTX_PUSH("ecb256_enc"); aes256_ecb_encrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("ctr128_enc"); aes128_ctr_encrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("ctr256_enc"); aes256_ctr_encrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isGcm && bits==128){ NVTX_PUSH("gcm128_enc"); aes128_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                    else if(isGcm && bits==256){ NVTX_PUSH("gcm256_enc"); aes256_gcm_encrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag); NVTX_POP(); }
                } else {
                    if(isEcb && bits==128){ NVTX_PUSH("ecb128_dec"); aes128_ecb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isEcb && bits==256){ NVTX_PUSH("ecb256_dec"); aes256_ecb_decrypt<<<grid,block>>>(d_in,d_out,nBlocks); NVTX_POP(); }
                    else if(isCtr && bits==128){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("ctr128_dec"); aes128_ctr_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isCtr && bits==256){ uint64_t lo,hi; packCtr(iv.data(),lo,hi); NVTX_PUSH("ctr256_dec"); aes256_ctr_decrypt<<<grid,block>>>(d_in,d_out,nBlocks,lo,hi); NVTX_POP(); }
                    else if(isGcm && bits==128){ NVTX_PUSH("gcm128_dec"); aes128_gcm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                    else if(isGcm && bits==256){ NVTX_PUSH("gcm256_dec"); aes256_gcm_decrypt<<<1,THREADS_PER_BLOCK>>>(d_in,d_out,nBlocks,d_iv,d_tag,d_tag); NVTX_POP(); }
                }
                cudaEventRecord(e); CHECK_CUDA(cudaEventSynchronize(e)); float ms=0.0f; cudaEventElapsedTime(&ms,s,e);
                double gib=(double)bytes/(double)(1ull<<30); double thr=gib/(ms/1000.0);
                printf("RESULT_GPU,%s,%zu,%d,%.3f,%.3f,%s\n", mode.c_str(), bytes, run, ms, thr, decrypt?"DEC":"ENC");
                std::ofstream fg("bench/thr_gpu.csv",std::ios::app); fg<<"RESULT_GPU,"<<mode<<","<<bytes<<","<<run<<","<<ms<<","<<thr<<","<<(decrypt?"DEC":"ENC")<<"\n";

                std::vector<uint8_t> host_in(bytes); CHECK_CUDA(cudaMemcpy(host_in.data(),d_in,bytes,cudaMemcpyDeviceToHost));
                const EVP_CIPHER* (*sel)();
                if(isEcb&&bits==128) sel=&EVP_aes_128_ecb; else if(isEcb&&bits==256) sel=&EVP_aes_256_ecb;
                else if(isCtr&&bits==128) sel=&EVP_aes_128_ctr; else if(isCtr&&bits==256) sel=&EVP_aes_256_ctr;
                else if(isGcm&&bits==128) sel=&EVP_aes_128_gcm; else sel=&EVP_aes_256_gcm;
                double cpu_thr = cpu_aes_throughput(host_in.data(), bytes, key.data(), bits, decrypt, sel);
                double ms_cpu = (double)bytes/(cpu_thr*(1ull<<30))*1000.0;
                printf("RESULT_CPU,%s,%zu,%d,%.3f,%.3f,%s\n", mode.c_str(), bytes, run, ms_cpu, cpu_thr, decrypt?"DEC":"ENC");
                std::ofstream fc("bench/thr_cpu.csv",std::ios::app); fc<<"RESULT_CPU,"<<mode<<","<<bytes<<","<<run<<","<<ms_cpu<<","<<cpu_thr<<","<<(decrypt?"DEC":"ENC")<<"\n";

                CHECK_CUDA(cudaFreeHost(h_in)); CHECK_CUDA(cudaFreeHost(h_out));
                CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out)); if(d_tag) CHECK_CUDA(cudaFree(d_tag)); if(d_iv) CHECK_CUDA(cudaFree(d_iv));
            }
        }
    }
    return 0;
}

/*
BUILD:
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

BASELINE RUN (all figures R-1..R-3):
build/CudaProject.exe

OCCUPANCY SWEEP (M-2, D-3):
for %B in (32 64 128 256 512) do build\CudaProject.exe --block %B

CTR COUNTER (I-4):
build/CudaProject.exe --ctr-preview

GHASH PARTIALS (M-3):
build/CudaProject.exe --gcm-debug

GF MULTIPLY (D-2):
build/CudaProject.exe --gf-mult

NSYS TIMELINE + MEM-TRACE (M-1 & M-4):
nsys profile --trace cuda,nvtx -o bench/run build/CudaProject.exe

ROOFLINE METRICS (M-5, D-1):
ncu --metrics flop_count_sp,dram__bytes_read.sum,dram__bytes_write.sum
--set full --csv --target-processes all build/CudaProject.exe

PTX DUMPS (D-5):
nvdisasm --print-code aes128_ecb.cu.obj > bench/ptx_lookup.txt
*/

# CUDA-AES Benchmark

CUDA-AES is a CUDA benchmark and research repository for AES kernels on NVIDIA GPUs. The current canonical implementation benchmarks AES-128 and AES-256 in ECB, CTR, and GCM-shaped code paths, with OpenSSL used as the CPU comparison path.

This repository is being prepared as an open-source GPU AES benchmark project. It is not production cryptography software.

## Current Status

- Canonical source lives in the top-level `.cu` and `.h` files built by the root `CMakeLists.txt`.
- `v3/` is a local experimental variant and is not the canonical Phase 1 build target.
- `cihangirTezcanAESimplementation/` is legacy/provenance code.
- GCM correctness and authentication are scheduled for Phase 2. Until NIST/OpenSSL known-answer tests pass, do not treat the current GCM output as standards-compliant AES-GCM.

## Prerequisites

- NVIDIA GPU with a CUDA-capable driver
- CUDA Toolkit with `nvcc` available
- CMake 3.28 or newer
- A CUDA-compatible host C++ compiler
- OpenSSL development package discoverable by CMake

On Windows, use a Visual Studio Developer Command Prompt or pass the host compiler explicitly with `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`.

If OpenSSL is installed in a custom location, pass `OPENSSL_ROOT_DIR`:

```bash
cmake -S . -B build -DOPENSSL_ROOT_DIR=/path/to/openssl
```

## Configure And Build

Choose the CUDA architecture for your GPU. For example, `86` targets Ampere GPUs such as RTX 30-series cards.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
```

For multi-config generators such as Visual Studio, `--config Release` selects the Release configuration at build time.

## Run

Linux:

```bash
./build/CudaProject
```

Windows:

```powershell
.\build\Release\CudaProject.exe
```

The benchmark currently exercises ECB, CTR, and GCM-shaped paths for AES-128 and AES-256 across several message sizes. Example output:

```text
[RUN 3/5] [GPU] ctr-128 processed 100 MiB in 12.3 ms -> 7.9 GiB/s
```

Use repeated runs, fixed GPU clocks, and a quiet system when comparing throughput numbers.

## Optional Tooling

Generate a PTX dump for `aes128_ecb.cu`:

```bash
cmake --build build --target ptx-dump
```

Run the Nsight Systems helper target by configuring the executable path first:

```bash
cmake -S . -B build -DNSYS_EXECUTABLE="/path/to/nsys"
cmake --build build --target nsight-profile
```

## Roadmap Direction

The near-term roadmap focuses on making this a credible open-source GPU AES benchmark:

- Portable build and repository hygiene
- Standard AES-GCM correctness with tag verification and known-answer tests
- Complete AES mode roadmap beyond the currently implemented modes
- Reproducible benchmark harness and published result format
- Documentation, contribution workflow, and SEO-ready project presentation

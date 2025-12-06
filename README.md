# CUDA-AES Benchmark

This project contains CUDA implementations of AES encryption kernels.

## CI/CD

This project uses GitHub Actions for continuous integration across multiple Ubuntu environments:
- Ubuntu 20.04 LTS
- Ubuntu 22.04 LTS
- Ubuntu 24.04 LTS

The CI pipeline automatically builds and validates the project on each push and pull request.

## Build

### Dependencies

- NVIDIA CUDA Toolkit (12.x or compatible)
- CMake (3.28 or higher)
- OpenSSL development libraries
  - On Ubuntu/Debian: `sudo apt-get install libssl-dev`
  - On Windows: Manual installation required

### Building the Project

Ensure the NVIDIA CUDA Toolkit is installed and `nvcc` is in your `PATH`.
Then build in Release mode using CMake:

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The resulting executable `CudaProject` will be generated in the `build` directory.

## Run

Enable persistent GPU mode (optional):

```bash
sudo nvidia-smi -pm 1
```

Run the benchmark:

```bash
./CudaProject
```

The program executes AES in ECB, CTR and GCM modes for both 128 and 256-bit
keys. It measures throughput for message sizes of 1 MB, 10 MB, 100 MB and 1 GB.
Each configuration is executed 5 times. Example output line:

```
[RUN 3/5] [GPU] ctr-128 processed 100 MiB in 12.3 ms -> 7.9 GiB/s
```

Use these lines to compute averages for your experiments.


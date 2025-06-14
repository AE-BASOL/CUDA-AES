# CUDA-AES Benchmark

This project contains CUDA implementations of AES encryption kernels.

## Build

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


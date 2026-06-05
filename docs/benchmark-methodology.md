# Benchmark Methodology

CUDA-AES benchmark results should be reproducible from raw artifacts, not copied from console output.

## Recommended Procedure

1. Build in Release mode and record the exact CMake command.
2. Run `ctest --test-dir build --output-on-failure`.
3. Run the benchmark with explicit parameters:

```bash
./build/CudaProject --runs 5 --sizes 1048576,10485760,104857600 --bench-dir bench/run-001
```

4. Record fixed clocks, persistence mode, GPU model, driver, CUDA Toolkit, OS, compiler, and command line.
5. Generate a summary:

```bash
python scripts/summarize_benchmarks.py bench/run-001/thr_gpu.csv bench/run-001/thr_cpu.csv -o bench/run-001/summary.md
```

6. Publish raw CSV files, `run_metadata.csv`, and `summary.md` together.

## Raw Output

The raw CSV schema is `phase3.v1`:

```text
schema_version,benchmark_run_id,timing_scope,device,cipher,block_size,run_index,run_count,time_ms,GiB/s,operation,command_line
```

`run_metadata.csv` records command, run count, selected sizes, CUDA versions, GPU properties, and environment hints.

## Timing Scope

GPU rows use `timing_scope=kernel_only`. This measures the CUDA kernel region with CUDA events and excludes allocation, transfers, validation, and summary generation.

CPU rows use `timing_scope=cpu_baseline`. They are OpenSSL EVP comparison rows, not a tuned CPU benchmark study.

Future end-to-end rows must use a distinct `timing_scope`.

## Authenticated Encryption Modes

GCM and CCM rows include authentication tag work in the measured kernel. Current CCM benchmark scope is 96-bit nonce, empty AAD, 16-byte tag, and full 16-byte payload blocks. These rows should not be interpreted as a complete AEAD API with arbitrary AAD, partial-block payloads, variable tag lengths, or variable nonce lengths.

## Storage Modes

XTS-AES rows model full-block storage data units with a 16-byte sector tweak and two AES key schedules. XTS is confidentiality-only and does not authenticate data. Current benchmark rows do not implement ciphertext stealing, so non-block-multiple storage sectors are out of scope.

## Key-Wrap Modes

AES-KW and AES-KWP rows are key-management workload rows, not bulk encryption throughput rows. The benchmark batches fixed-size records so the existing CSV schema can still be used:

- AES-KW wraps 16-byte key-data records to 24-byte wrapped records.
- AES-KWP wraps 20-byte key-data records to 32-byte wrapped records.
- `block_size` records the total input key-material bytes processed across the batch.
- Current key-wrap benchmark rows are GPU `kernel_only` rows; CPU baseline rows are not emitted for AES-KW/AES-KWP yet.

## Feedback Modes

CBC, CFB, and OFB are feedback modes with per-message dependency chains. Their benchmark rows are useful for reproducibility and mode coverage, but they should not be read as naturally parallel CTR-like throughput. CBC encryption in particular is dependency-bound; CBC decryption can expose more block-level parallelism because each plaintext block depends on the current and previous ciphertext blocks. CFB rows use full-block CFB-128 segment semantics.

## Limitations

- Do not compare kernel-only and end-to-end rows as the same metric.
- Do not publish throughput claims without raw artifacts.
- Do not treat the project as production cryptography software.
- Interpret CBC, CFB, and OFB feedback-mode throughput with their dependency chains in mind.
- GCM benchmark scope matches the current correctness scope: 96-bit IV, empty AAD, full blocks.
- CCM benchmark scope matches the current correctness scope: 96-bit nonce, empty AAD, 16-byte tag, full blocks.
- XTS-AES benchmark scope is full 16-byte blocks with a 16-byte sector tweak; ciphertext stealing is out of scope.
- AES-KW/AES-KWP benchmark scope is fixed-size batched key-wrap records, not streaming buffers.

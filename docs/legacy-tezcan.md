# Legacy Tezcan Implementation

The `cihangirTezcanAESimplementation/` directory preserves the original GPU AES implementation notes and code lineage that motivated this repository. It is useful provenance, but it is not the canonical build target for the current CUDA-AES Benchmark project.

## Provenance

The legacy README describes CUDA optimizations of a T-table AES implementation with zero bank conflicts and reports published GPU AES-128 encryption throughput results:

- 315.2 Gbps on GTX 970
- 878.6 Gbps on RTX 2070 Super

The README attributes the work to Cihangir Tezcan and links the associated IEEE publication. It also explains the reproducibility motivation: public GPU AES optimization code is rare, and comparing optimizations across different GPUs is difficult unless implementations can be run on the same hardware.

## Status In This Repository

For this repository, the canonical implementation is the top-level CMake target and top-level source files such as `main.cu`, `aes_common.h`, `aes_tables.cu`, and `aes128_*.cu` / `aes256_*.cu`.

The legacy folder is kept as reference and provenance material. It should not be treated as current project documentation, current benchmark methodology, or evidence for the top-level benchmark results unless a future phase explicitly integrates and verifies it.

## License And Citation Note

The Phase 4 repository license and citation metadata apply to the current public CUDA-AES Benchmark package. The legacy folder's original attribution should remain visible when discussing or modifying that material. Do not remove the Tezcan attribution or imply that the legacy performance numbers were produced by the current benchmark harness.

## Practical Guidance

- Use the top-level README and `docs/` pages for current build, correctness, benchmark, and contribution instructions.
- Use the legacy folder to understand original motivation and related published work.
- Use current raw benchmark artifacts for new performance claims; do not carry legacy throughput numbers into current result summaries without reproducing them through the current methodology.

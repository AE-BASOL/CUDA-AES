# AES Mode Matrix

This project tracks the practical AES mode family so readers can see what is implemented today, what is tested, what is benchmarked, and what remains planned work.

The canonical implementation is the top-level CMake target and top-level CUDA/C++ source files. Legacy and experimental folders are documented separately and are not counted as implemented coverage in this matrix.

## Status Matrix

| Mode | Implemented | Correctness tests | Benchmark rows | Documentation status | Notes / phase |
|------|-------------|-------------------|----------------|----------------------|---------------|
| ECB | Yes | Yes | Yes | Current docs | Implemented for AES-128 and AES-256. Covered by deterministic known-answer tests and benchmark rows. |
| CBC | Yes | Yes | Yes | Current docs | Implemented for AES-128 and AES-256. Confidentiality-only feedback mode from NIST SP 800-38A. |
| CFB | Yes | Yes | Yes | Current docs | Implemented for AES-128 and AES-256 with CFB-128 full-block segment semantics only. |
| OFB | Yes | Yes | Yes | Current docs | Implemented for AES-128 and AES-256. Confidentiality-only stream-like mode from NIST SP 800-38A. |
| CTR | Yes | Yes | Yes | Current docs | Implemented for AES-128 and AES-256. Benchmark helper currently uses a 96-bit IV/counter convention. |
| GCM / GMAC | GCM: Yes; GMAC: No | GCM: Yes; GMAC: No | GCM: Yes; GMAC: No | Current GCM docs; GMAC planned | GCM is implemented for the current benchmark scope: 96-bit IV, empty AAD, and full 16-byte blocks. GMAC is authentication-only and should not be mixed with encryption throughput. |
| CCM | No | No | No | Planned | Phase 6 target. Authenticated encryption mode; nonce, tag length, and payload-size parameters must be documented with benchmark rows. |
| XTS-AES | No | No | No | Planned | Phase 6 target. Storage-oriented mode; benchmark parameters should describe sector and tweak handling. |
| AES-KW | No | No | No | Planned | Phase 6 target. Key-wrap workload; benchmark shape differs from bulk encryption throughput. |
| AES-KWP | No | No | No | Planned | Phase 6 target. Key-wrap with padding; benchmark shape differs from bulk encryption throughput. |

## Boundary Notes

- ECB, CBC, CFB, OFB, and CTR are confidentiality modes. They do not provide authentication.
- CBC, CFB, and OFB have feedback dependencies. Their benchmark rows are included for reproducible mode coverage, not as evidence of CTR-like natural parallelism.
- CFB coverage is CFB-128 only. Smaller CFB segment sizes are future work if they become useful for benchmarking.
- GCM and CCM are authenticated encryption modes. Benchmark rows must preserve tag, nonce, AAD, and message-size assumptions.
- GMAC and CMAC are authentication/MAC workloads, not bulk encryption modes. Future GMAC or CMAC benchmark rows should be labeled as authentication throughput so they are not compared directly with encryption throughput.
- AES-KW and AES-KWP are key-wrap modes. They should be evaluated with key-wrap-sized payloads instead of only large streaming buffers.

## Current Reader Takeaway

The repo currently implements, tests, and benchmarks ECB, CBC, CFB-128, OFB, CTR, and GCM for AES-128 and AES-256 in the canonical build. CCM, XTS-AES, AES-KW, AES-KWP, GMAC, and CMAC are roadmap items unless later phases update this matrix.

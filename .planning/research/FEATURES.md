# Research: Features

## Question

What features should v1 include for a high-trust, discoverable GPU AES benchmark repository?

## Table Stakes

- Clear one-line value proposition in README: CUDA AES benchmark suite with reproducible GPU throughput results.
- Build instructions for Windows and Linux, or explicit supported-platform boundaries.
- Correctness tests using known AES vectors for ECB, CTR, and GCM.
- Mode expansion plan for CBC, CFB, OFB, CCM, XTS-AES, AES-KW, and AES-KWP.
- Reproducible benchmark command that captures hardware/software environment.
- Results table with GPU, CUDA version, driver, OS, compiler, modes, sizes, runs, mean throughput, and variance where available.
- Raw benchmark output committed or attached to releases.
- License, contributing guide, security policy, citation metadata, and changelog.
- GitHub topics and description targeting likely searches: `cuda`, `aes`, `gpu`, `benchmark`, `cryptography`, `gpgpu`, `openssl`, `nvidia`.

## Differentiators

- Separate kernel-only and end-to-end throughput.
- Include Nsight/NVTX profiling walkthrough.
- Publish a methodology page explaining fairness, warmup, clock mode, persistence mode, and measurement limitations.
- Add badges for build/test status, license, release, and citation.
- Add diagrams explaining host/device flow and AES mode coverage.
- Include a mode matrix showing status per mode: planned, implemented, tested, benchmarked, documented.
- Add GitHub Pages documentation for better search indexing and shareability.
- Add benchmark result artifacts per release with machine-readable CSV/JSON.

## Anti-Features

- Do not publish unsupported "fastest" claims without comparable datasets.
- Do not imply production cryptographic safety before API design, authentication semantics, and tests are mature.
- Do not call the project "all AES modes" until each mode has tests, benchmark data, and documentation.
- Do not hide build constraints or only support the author's local machine.
- Do not bury benchmark methodology below flashy throughput numbers.

## Sources

- GitHub repository best practices: https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories
- Google SEO Starter Guide: https://developers.google.cn/search/docs/fundamentals/seo-starter-guide?hl=en
- Open Mainframe repository best practices: https://tac.openmainframeproject.org/best_practices/repo.html

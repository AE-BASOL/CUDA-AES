# Research Summary

## Stack

Keep CUDA C++, CMake, CUDA Toolkit, OpenSSL, and NVIDIA profiling support. The important change is not a new stack; it is making the existing stack portable, documented, testable, and repeatable.

## Table Stakes

- Portable build.
- Correctness tests.
- Reproducible benchmark script.
- Environment capture.
- Clear benchmark methodology.
- Raw and summarized results.
- Roadmap coverage for full practical AES mode family, not only current ECB/CTR/GCM.
- Strong README.
- License, contributing guide, security policy, citation file, changelog, issue templates, and PR template.
- GitHub metadata and searchable docs.

## Watch Out For

- Do not lead with performance claims before correctness and reproducibility are credible.
- Do not let absolute local paths survive public release.
- Do not market as a production crypto library in v1.
- Do not treat SEO as keywords only; build real, crawlable, useful technical content.

## Recommended V1 Shape

V1 should be a public-ready benchmark repository, not a library. It should give CUDA developers a clean path:

1. Clone.
2. Build.
3. Run correctness tests.
4. Run benchmarks.
5. Inspect methodology.
6. Compare results.
7. Cite or contribute.

## Mode Coverage Direction

Use the NIST 800-38 family as the reference frame:

- Current code: ECB, CTR, GCM.
- Next confidentiality modes: CBC, CFB, OFB.
- Next authenticated/specialized modes: CCM, XTS-AES, AES-KW, AES-KWP.
- Related but separate track: GMAC/CMAC as authentication/MAC benchmarking rather than bulk encryption throughput.

# Roadmap

## Phase 9: Repository Hygiene and Branch Cleanup
**Goal**: Resolve and prune all stale remote and local feature branches to improve repository maintainability and presentation.
**Requirements**: MAINT-05, MAINT-06
**Success Criteria**:
1. All stale remote branches are deleted.
2. GitHub repository settings are verified to be aligned with open-source best practices.

## Phase 10: GPU Crypto Ecosystem Research
**Goal**: Research the GPU cryptography ecosystem to evaluate competitor solutions, mode completeness, and propose new features.
**Requirements**: RES-01, RES-02
**Success Criteria**:
1. A research document comparing the repository with existing GPU cryptography solutions.
2. At least one new feature or cryptographic mode is proposed and documented for implementation.

## Phase 11: CPU vs GPU Benchmark Framework
**Goal**: Implement a benchmark comparison framework using OpenSSL for CPU reference.
**Requirements**: BENCH-10, BENCH-11
**Success Criteria**:
1. Benchmark tool can run and measure OpenSSL CPU AES implementations.
2. Results table outputs both GPU and CPU performance metrics for comparison.

## Phase 12: Global Packaging and Distribution
**Goal**: Transform the repository into a globally accessible package (e.g., `npx`) to maximize reach and usability.
**Requirements**: PKG-01, PKG-02
**Success Criteria**:
1. The project can be executed globally using `npx` or a similar tool without local compilation overhead.
2. README is updated with instructions on how to use the global package.

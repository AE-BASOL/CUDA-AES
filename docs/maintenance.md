# Maintenance

This document defines the post-v1 maintenance loop for CUDA-AES Benchmark. It keeps the repository focused on reproducible CUDA AES benchmarking rather than production cryptography library support.

## Triage Categories

Use these categories when reviewing issues and pull requests:

- `build`: CMake, CUDA Toolkit, host compiler, OpenSSL, or platform setup issues.
- `correctness`: deterministic known-answer test failures, authentication/tag behavior, or mode semantics.
- `benchmark result`: contributed result packages, raw artifacts, summary generation, timing-scope interpretation, and environment metadata.
- `documentation`: README, docs pages, claims policy, mode matrix, citation, changelog, and governance updates.
- `security`: correctness or documentation issues with security-sensitive impact.
- `mode request`: future AES mode, MAC/authentication workload, or benchmark coverage proposals.

## Changelog Loop

Keep `CHANGELOG.md` current:

1. Add user-visible changes to the `Unreleased` section as work lands.
2. Group changes by type, such as Added, Changed, Fixed, Documentation, or Known Verification Debt.
3. Move `Unreleased` entries into a dated release section only when the release gate passes.
4. Start a fresh `Unreleased` section after each versioned release.

Do not use the changelog for speculative future work. Put future work in the roadmap instead.

## Roadmap Loop

Update `.planning/ROADMAP.md`, README roadmap direction, or follow-up planning artifacts when:

- supported AES mode scope changes;
- benchmark artifact schema or timing scope changes;
- release verification requirements change;
- v1 maintenance work becomes v2 scope;
- a new feature would imply production library or API guarantees.

Small bug fixes do not need roadmap churn unless they change public scope or maintenance direction.

## Benchmark Result Review

Review benchmark result contributions against `docs/benchmark-result-contributions.md`:

1. Confirm the commit hash and CTest status.
2. Confirm configure/build and benchmark commands are present.
3. Confirm `run_metadata.csv`, `thr_gpu.csv`, `thr_cpu.csv` where applicable, and generated `summary.md` are attached or linked.
4. Check GPU model, compute capability, CUDA Toolkit, driver, OS, compiler, clocks, and persistence-mode notes.
5. Verify claims remain environment-specific and raw-artifact-backed.
6. Reject or reframe ranking claims unless controlled comparative evidence is included.

Contributed results can be useful as environment-specific evidence without becoming official release artifacts.

## Security Handling

For security-sensitive reports, prefer GitHub private vulnerability reporting when it is enabled. If it is not enabled, ask the reporter to use the clearest maintainer contact path available and not to disclose sensitive details in public issues.

Security triage should preserve the project scope:

- accept reports that affect benchmark correctness, authentication/tag behavior, misleading safety claims, or reproducibility;
- avoid implying production cryptography API guarantees;
- keep sensitive reproduction details out of public issue threads until the maintainer chooses an appropriate disclosure path.

## Future v2 Boundary

Keep v2 work separate from v1 maintenance unless the roadmap changes. Examples of v2 work:

- GMAC and CMAC benchmarking as standalone authentication/MAC workloads;
- benchmark charts and multi-GPU result tables;
- benchmark matrix automation across CUDA versions and architectures;
- GitHub Pages or a standalone project website;
- DOI-backed archived releases;
- paper-style technical reports;
- production library or API packaging.

These are valid future directions, but they should not silently expand the v1 maintenance scope.

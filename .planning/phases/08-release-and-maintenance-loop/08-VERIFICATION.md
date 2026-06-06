---
phase: 08-release-and-maintenance-loop
status: passed_with_environment_limit
verified_at: 2026-06-06T21:21:00+03:00
requirements: [REPO-04, MAINT-01, MAINT-02, MAINT-03, MAINT-04]
release_gate: verification-blocked
---

# Phase 8 Verification: Release And Maintenance Loop

## Result

Phase 8 source-controlled release and maintenance artifacts pass. The final runtime release gate is `verification-blocked` in this shell because `nvcc` cannot find the Visual Studio host compiler `cl.exe` during Release CMake configuration.

Do not publish the final `v1.0.0` GitHub Release as complete from this environment. Use `docs/release-v1.md` as a release candidate handoff until the release gate records `verification-passed`.

## Requirement Evidence

| Requirement | Status | Evidence |
|-------------|--------|----------|
| REPO-04 | Release candidate ready; final publication blocked by runtime environment | `docs/release-v1.md` defines the `v1.0.0` release tag/title, release notes, build/test status, verification commands, benchmark environment fields, raw artifact manifest, known limitations, maintenance/security links, and publication checklist. |
| MAINT-01 | Passed | Markdown issue and pull request templates guide useful bug reports, mode requests, benchmark-result issues, and PRs. Templates include GitHub chooser frontmatter where applicable. |
| MAINT-02 | Passed | `docs/benchmark-result-contributions.md` defines the canonical benchmark result checklist with commit hash, CTest status, configure/build command, benchmark command, raw CSV artifacts, generated `summary.md`, environment metadata, clocks/persistence note, and claims acknowledgement. |
| MAINT-03 | Passed with release gate caveat | `docs/maintenance.md` defines changelog, roadmap, triage, benchmark review, security, and v2-boundary loops. `CHANGELOG.md` preserves a fresh `Unreleased` section and release-candidate verification debt instead of prematurely promoting a final `1.0.0` section. |
| MAINT-04 | Passed | `SECURITY.md` prefers GitHub private vulnerability reporting when enabled and gives a safe public fallback that does not include sensitive details. |

## Release Gate Attempt

Attempted:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
```

Result:

```text
-- Building for: Ninja
-- If nvcc cannot find cl.exe, run CMake from a Visual Studio Developer Command Prompt or pass -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>.
-- The CXX compiler identification is GNU 13.2.0
-- Configuring incomplete, errors occurred!
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

Because configure failed before build files were generated for CUDA, these commands were not run:

```powershell
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke
python scripts\summarize_benchmarks.py bench\v1-smoke\thr_gpu.csv bench\v1-smoke\thr_cpu.csv -o bench\v1-smoke\summary.md
```

## Release Candidate Handoff

To close the release gate, run from a Visual Studio Developer Command Prompt or pass an explicit host compiler:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke
python scripts\summarize_benchmarks.py bench\v1-smoke\thr_gpu.csv bench\v1-smoke\thr_cpu.csv -o bench\v1-smoke\summary.md
```

If all commands pass:

- update this file's `release_gate` to `verification-passed`;
- record the exact commands, commit hash, environment fields, and artifact paths;
- promote `CHANGELOG.md` to a dated `## [1.0.0] - YYYY-MM-DD` section with a fresh `## Unreleased` section;
- publish or save the GitHub Release draft for tag `v1.0.0`;
- attach only raw artifacts from the verified release-gate run.

## Claim Guardrails

No benchmark numbers were added. No stale benchmark artifacts were attached or listed as verified release assets. Release wording remains environment-specific and raw-artifact-backed, with no fastest-in-world, universal ranking, or production cryptography library claim.

## Verification Commands

- `Test-Path docs\release-v1.md` - passed.
- `rg "v1.0.0|release candidate|CTest|run_metadata.csv|thr_gpu.csv|thr_cpu.csv|summary.md|known limitations|benchmark-result-contributions.md|maintenance.md" docs\release-v1.md` - passed.
- `Test-Path .planning\phases\08-release-and-maintenance-loop\08-VERIFICATION.md` - passed.
- `rg "REPO-04|MAINT-03|release gate|ctest|CudaProject|summary.md|verification-passed|verification-blocked|release candidate" .planning\phases\08-release-and-maintenance-loop\08-VERIFICATION.md docs\release-v1.md` - passed.
- `rg "Unreleased|1.0.0|Known Verification Debt|release candidate|cl.exe|CTest" CHANGELOG.md` - passed.
- Guardrail check: no unsupported ranking claim was added to `docs/release-v1.md`.

## Verification Complete

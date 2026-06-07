---
phase: 08
slug: release-and-maintenance-loop
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-06
---

# Phase 08 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | PowerShell / ripgrep / existing CMake, CTest, and benchmark commands |
| **Config file** | `CMakeLists.txt` |
| **Quick run command** | `rg "Benchmark result contribution|private vulnerability|v1.0.0|release candidate|CHANGELOG|roadmap" CONTRIBUTING.md SECURITY.md docs .github .planning/phases/08-release-and-maintenance-loop` |
| **Full suite command** | `ctest --test-dir build --output-on-failure` plus smoke benchmark when CUDA host compiler environment is available |
| **Estimated runtime** | Documentation checks under 10 seconds; runtime release gate depends on CUDA build and benchmark environment |

---

## Sampling Rate

- **After every task commit:** Run the task-specific `rg` / `Test-Path` checks.
- **After every plan wave:** Run the plan-level verification block from each completed plan.
- **Before `$gsd-verify-work`:** Run all documentation checks, then run the runtime release gate if a CUDA host compiler environment is available.
- **Max feedback latency:** Documentation checks under 10 seconds.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | MAINT-02 | T-BENCH-01 | Benchmark contributions require raw artifacts and environment metadata | docs | `Test-Path docs/benchmark-result-contributions.md` | yes | pending |
| 08-01-02 | 01 | 1 | MAINT-01 | T-BENCH-02 | Contributor entry points route to one checklist | docs | `rg "docs/benchmark-result-contributions.md|Benchmark result contribution" CONTRIBUTING.md docs/results.md .github` | yes | pending |
| 08-02-01 | 02 | 1 | MAINT-04 | T-SEC-01 | Security reports have a concrete non-public path where available | docs | `rg "private vulnerability reporting|not include sensitive|benchmark and research software" SECURITY.md` | yes | pending |
| 08-02-02 | 02 | 1 | MAINT-03 | T-MAINT-01 | Changelog, roadmap, issue triage, and security review loop are documented | docs | `Test-Path docs/maintenance.md; rg "CHANGELOG|roadmap|triage|security|benchmark result" docs/maintenance.md` | yes | pending |
| 08-03-01 | 03 | 2 | REPO-04 | T-REL-01 | Release notes include verification gate, artifact manifest, and limitations | docs | `Test-Path docs/release-v1.md; rg "v1.0.0|release candidate|CTest|run_metadata.csv|known limitations" docs/release-v1.md` | yes | pending |
| 08-03-02 | 03 | 2 | REPO-04 | T-REL-02 | Final v1 is blocked unless runtime verification passes | manual/runtime | `ctest --test-dir build --output-on-failure` | yes | pending |

---

## Wave 0 Requirements

Existing infrastructure covers all Phase 8 documentation checks. Runtime verification depends on an environment where `nvcc` can find `cl.exe`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final v1 runtime release gate | REPO-04 | Current normal shell cannot prove CUDA host compiler setup | Run the release-gate commands from a Visual Studio Developer Command Prompt or with `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>` and paste the output into the Phase 8 execution checkpoint. |
| GitHub Release publication | REPO-04 | Requires repository write permissions and maintainer intent | Review `docs/release-v1.md`, attach only verified artifacts, and publish or save the GitHub Release draft. |
| Enable private vulnerability reporting | MAINT-04 | Requires repository admin settings | Enable GitHub private vulnerability reporting if available, or leave `SECURITY.md` fallback instructions in place. |

---

## Validation Sign-Off

- [x] All tasks have automated documentation checks or explicit manual gate coverage.
- [x] Sampling continuity: no 3 consecutive tasks without automated verification.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Documentation feedback latency under 10 seconds.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** pending execution

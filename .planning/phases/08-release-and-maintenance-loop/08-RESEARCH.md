# Phase 8: Release And Maintenance Loop - Research

**Researched:** 2026-06-06
**Status:** Complete

## Research Question

What does the executor need to know to plan a credible v1 release and maintenance loop for CUDA-AES Benchmark without weakening benchmark credibility or implying production cryptography safety?

## Phase Scope

Phase 8 covers REPO-04, MAINT-01, MAINT-02, MAINT-03, and MAINT-04:

- versioned release preparation;
- release notes and artifact policy;
- issue and pull request contribution surfaces;
- benchmark result contribution checklist;
- changelog, roadmap, and maintenance loop;
- security reporting path and benchmark-vs-library scope.

It should not add AES modes, change benchmark semantics, publish unsupported throughput claims, introduce GitHub Pages, or turn the project into a production cryptography library.

## Current State

- `CHANGELOG.md` has an `Unreleased` section and a known verification debt note.
- `CONTRIBUTING.md` already lists build, CTest, benchmark, metadata, and raw artifact expectations.
- `.github/ISSUE_TEMPLATE/benchmark_result.md`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/mode_request.md`, and `.github/pull_request_template.md` exist.
- `SECURITY.md` already states benchmark/research scope but has an imprecise reporting path.
- `docs/results.md` and `docs/benchmark-methodology.md` define result package contents, raw CSV schema, timing scope, and claims policy.
- Runtime CMake/CTest/benchmark verification is still blocked in the normal shell until `nvcc` can find `cl.exe`.

## External Findings

### GitHub Releases

GitHub Releases are tag-based release records that can include release notes and attached files. GitHub's release UI and CLI support creating a new tag, entering release notes, attaching binaries or assets, marking prereleases, and saving drafts before publishing. Draft-first release preparation is especially relevant when assets must be attached before publication.

Sources:
- GitHub Docs, Managing releases in a repository: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository
- GitHub Docs, About releases: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

Implementation implication: v1 should be planned as a `v1.0.0` GitHub Release draft with release notes and optional verified artifact attachments. The source tree should contain release notes or a release checklist so publication does not depend on memory or GitHub UI-only text.

### Versioning

Semantic Versioning uses `MAJOR.MINOR.PATCH`; `1.0.0` defines the public API. For this repository, the public contract is not a production cryptography API. It is the public benchmark-suite contract: documented build, correctness checks, benchmark artifact schema, supported mode scope, release artifact expectations, and claims policy.

Source:
- Semantic Versioning 2.0.0: https://semver.org/

Implementation implication: use `v1.0.0` as the release identifier, but be explicit that it stabilizes the benchmark/reproducibility contract, not a library API guarantee.

### Changelog

Keep a Changelog recommends a human-readable `CHANGELOG.md`, a top `Unreleased` section, grouped change categories, latest version first, release dates, and moving `Unreleased` entries into a version section at release time.

Source:
- Keep a Changelog 1.1.0: https://keepachangelog.com/en/1.1.0/

Implementation implication: `CHANGELOG.md` should become release-ready with a `v1.0.0` section only after the release verification gate passes. If verification remains blocked, keep the release as a candidate and record the blocker rather than pretending the release is complete.

### Issue And PR Templates

GitHub supports issue and pull request templates. Markdown issue templates in `.github/ISSUE_TEMPLATE` should include frontmatter to appear cleanly in the template chooser. Issue forms are available, but the Phase 8 context explicitly keeps Markdown templates for v1.

Source:
- GitHub Docs, About issue and pull request templates: https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates

Implementation implication: keep the existing Markdown templates, add or tighten YAML frontmatter, and link detailed benchmark-result requirements to one canonical checklist document.

### Private Vulnerability Reporting

GitHub private vulnerability reporting lets security researchers privately report vulnerabilities to maintainers when enabled. If it is not enabled, reporters must follow the repository's `SECURITY.md` instructions or open a public issue asking for the preferred contact path without disclosing sensitive details.

Sources:
- GitHub Docs, Configuring private vulnerability reporting: https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository
- GitHub Docs, Privately reporting a security vulnerability: https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately

Implementation implication: `SECURITY.md` should prefer GitHub private vulnerability reporting when enabled and provide a safe fallback path that does not ask reporters to post sensitive details publicly.

## Planning Recommendation

Use three plans:

1. Benchmark contribution intake and canonical checklist.
2. Security policy and maintenance loop.
3. v1 release-candidate package and release verification gate.

Plans 1 and 2 can run in parallel because they touch disjoint files. Plan 3 should depend on both because the release package should point at the finalized contribution, security, and maintenance surfaces.

## Validation Architecture

### Automated Validation

Use file-existence and `rg` checks for documentation/guidance changes:

- canonical benchmark checklist exists and is linked from contribution entry points;
- issue templates include useful frontmatter and required fields;
- `SECURITY.md` includes benchmark-vs-library scope plus private-reporting guidance;
- maintenance loop doc exists and describes changelog, roadmap, triage, benchmark review, and security handling;
- v1 release notes/checklist includes build/test status, benchmark environment, raw artifacts, known limitations, and the release gate.

### Runtime Release Gate

The final release plan must require a real runtime verification pass:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke
python scripts\summarize_benchmarks.py bench\v1-smoke\thr_gpu.csv bench\v1-smoke\thr_cpu.csv -o bench\v1-smoke\summary.md
```

If the environment still cannot run CUDA verification, the output must remain a release-candidate checklist and handoff. Do not publish final v1 claims or attach stale benchmark artifacts.

## Research Complete

## RESEARCH COMPLETE

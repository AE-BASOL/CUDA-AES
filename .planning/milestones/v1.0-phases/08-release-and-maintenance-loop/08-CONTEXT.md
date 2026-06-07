# Phase 8: Release And Maintenance Loop - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 publishes a credible v1 release path and defines how the repository stays maintainable after v1. It covers release notes, release gating, changelog readiness, benchmark result packaging and review, issue/PR guidance, and security reporting scope.

This phase does not add new AES modes, change benchmark semantics, add GitHub Pages, make production cryptography guarantees, or introduce unsupported performance claims.

</domain>

<decisions>
## Implementation Decisions

### Release Package
- **D-01:** Use a SemVer-style `v1.0.0` tag and GitHub Release as the canonical v1 release surface unless the repository owner has a different explicit versioning policy.
- **D-02:** v1 release notes must include build/test status, benchmark environment, exact commands, raw artifact manifest, generated summary reference, known limitations, and carried verification debt.
- **D-03:** Attach raw benchmark artifacts only from a verified run. If the verification environment is unavailable, prepare the release notes and artifact checklist but do not attach stale or unverified throughput files.
- **D-04:** Promote `CHANGELOG.md` from `Unreleased` into a dated `v1.0.0` section and keep a fresh `Unreleased` section for future maintenance.

### Release Gate
- **D-05:** Do not publish the final v1 release as complete until a runtime verification pass succeeds in a shell where `nvcc` can find `cl.exe`, or with an explicit `CMAKE_CUDA_HOST_COMPILER`.
- **D-06:** The release verification pass should include Release configure/build, `ctest --test-dir build --output-on-failure`, a smoke benchmark with an isolated `--bench-dir`, and summary generation from raw CSV files.
- **D-07:** If verification remains blocked, the deliverable should be a release-candidate checklist or handoff, not a final v1 release claim. Any public notes must state the blocker plainly.
- **D-08:** Benchmark claims remain environment-specific measurements tied to raw artifacts and timing scope. Do not introduce ranking or fastest-in-world language.

### Benchmark Contribution Checklist
- **D-09:** Create one canonical benchmark result contribution checklist document and link it from `CONTRIBUTING.md`, `.github/pull_request_template.md`, `.github/ISSUE_TEMPLATE/benchmark_result.md`, and `docs/results.md`.
- **D-10:** Keep the existing Markdown issue and PR templates for v1. Tighten them as short entry points to the canonical checklist rather than introducing GitHub issue forms in this phase.
- **D-11:** The checklist must require commit hash, CTest status, configure/build command, benchmark command, `run_metadata.csv`, `thr_gpu.csv`, `thr_cpu.csv` where applicable, generated `summary.md`, GPU model, compute capability, CUDA Toolkit, driver, OS, compiler, CMake options, clocks/persistence note, and claims-policy acknowledgement.
- **D-12:** Benchmark result contributions should be accepted as reproducible environment-specific evidence, not as universal performance rankings.

### Maintenance Loop
- **D-13:** Add a compact maintenance document or maintenance section that defines the post-v1 loop: keep `CHANGELOG.md` current, update roadmap/README direction when scope changes, and triage issues by build, correctness, benchmark, documentation, security, and mode-request categories.
- **D-14:** Security reporting guidance must be concrete. Prefer GitHub private vulnerability reporting if enabled; otherwise tell reporters not to include sensitive data in public issues and provide the clearest available maintainer contact path.
- **D-15:** Keep v2 work separate from v1 maintenance. GMAC/CMAC benchmarking, charts, benchmark matrix automation, GitHub Pages, DOI-backed archived releases, paper-style reports, and library/API work remain future phases unless the roadmap changes.

### the agent's Discretion
- Exact filename for the new benchmark checklist is flexible if it is stable and linked from all relevant entry points.
- Exact release note wording and artifact bundle naming are flexible, constrained by the verification and claims-policy decisions above.
- The planner may decide whether the maintenance loop lives in a new `docs/maintenance.md` file or in an existing governance document, as long as the maintainer workflow is easy to find.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning
- `.planning/PROJECT.md` - project positioning, credibility constraints, active maintenance requirements, and verification debt.
- `.planning/REQUIREMENTS.md` - Phase 8 requirements `REPO-04`, `MAINT-01`, `MAINT-02`, `MAINT-03`, and `MAINT-04`.
- `.planning/ROADMAP.md` - fixed Phase 8 boundary and success criteria.
- `.planning/STATE.md` - current project state and carried verification debt.

### Prior Decisions
- `.planning/phases/01-repository-and-build-foundation/01-CONTEXT.md` - benchmark/research positioning, build portability expectations, and `cl.exe` verification context.
- `.planning/phases/07-discoverability-and-seo/07-CONTEXT.md` - focused benchmark metadata, conservative claims, stable docs links, and deferral of GitHub Pages.

### Codebase Context
- `.planning/codebase/CONVENTIONS.md` - documentation conventions, output patterns, and build/testing conventions.
- `.planning/codebase/STRUCTURE.md` - repository layout, docs folder, GitHub templates, and canonical source boundary.
- `.planning/codebase/BENCHMARKING.md` - benchmark artifact schema, correctness prerequisite, methodology, and known limitations.

### Public Release And Maintenance Files
- `README.md` - public landing page, benchmark scope, limitations, docs links, and roadmap direction.
- `CHANGELOG.md` - current `Unreleased` content and known verification debt to promote into v1 notes.
- `CONTRIBUTING.md` - existing contributor guidance and benchmark result contribution requirements.
- `SECURITY.md` - current benchmark-vs-library scope and reporting guidance needing concrete release-ready wording.
- `.github/pull_request_template.md` - current PR verification and benchmark artifact prompts.
- `.github/ISSUE_TEMPLATE/benchmark_result.md` - current benchmark result contribution template.
- `.github/ISSUE_TEMPLATE/bug_report.md` - current bug report environment and artifact prompts.
- `.github/ISSUE_TEMPLATE/mode_request.md` - current future-mode request template.
- `docs/results.md` - result package requirements and claims policy.
- `docs/benchmark-methodology.md` - reproducible benchmark procedure, raw output schema, and timing-scope rules.
- `docs/correctness.md` - correctness gate and verification limits.
- `docs/README.md` - stable documentation landing page and scope guardrails.
- `CITATION.cff` - project citation identity and release metadata context.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CHANGELOG.md`: already has an `Unreleased` section and known verification debt entry; use it as the source for v1 release-note structure.
- `CONTRIBUTING.md`: already captures the main benchmark result requirements; use it as seed text for the canonical checklist.
- `.github/ISSUE_TEMPLATE/benchmark_result.md`: already asks for environment, correctness gate, benchmark command, and raw artifacts.
- `.github/pull_request_template.md`: already includes verification and benchmark artifact sections.
- `SECURITY.md`: already states benchmark/research scope, but the reporting path is too vague for a release-ready repo.
- `docs/results.md` and `docs/benchmark-methodology.md`: already define credible result packages, raw CSV schema, timing scope, and claims policy.
- `scripts/summarize_benchmarks.py`: existing summary generator referenced by release verification and result package docs.

### Established Patterns
- Public docs are Markdown files linked from `README.md` and `docs/README.md`; keep this simple pattern.
- The project favors conservative benchmark claims tied to raw artifacts and environment metadata.
- Existing templates are Markdown files, not GitHub issue forms.
- Runtime verification is currently blocked in the ordinary shell by CUDA host compiler setup, and docs already name the Developer Command Prompt or explicit host compiler path as the fix.

### Integration Points
- Release notes should align `CHANGELOG.md`, `README.md`, `docs/results.md`, and `docs/benchmark-methodology.md` rather than inventing a separate claims policy.
- The benchmark checklist should become the target linked by `CONTRIBUTING.md`, PR template, benchmark result issue template, and results docs.
- Security guidance must preserve the existing benchmark-vs-production-library scope while giving reporters a concrete path.
- Roadmap/changelog maintenance should keep v2 ideas visible without pulling them into the v1 release scope.

</code_context>

<specifics>
## Specific Ideas

- Treat final v1 publication as blocked until the maintainer can run verification in a proper CUDA host compiler environment.
- If verification cannot run during implementation, produce a release-candidate checklist and handoff rather than publishing a final v1 release.
- Make `docs/results.md` and `docs/benchmark-methodology.md` the authority for raw artifact requirements and claim limits.
- Keep existing issue templates and PR template, but remove duplicated checklist details where a link to the canonical checklist is clearer.

</specifics>

<deferred>
## Deferred Ideas

- GitHub Pages or a standalone project website remains a future phase if repository docs are not enough.
- DOI-backed archived release, full paper-style report, and formal publication workflow remain v2 publication work.
- GMAC/CMAC benchmarking, benchmark charts, matrix automation across GPUs/CUDA versions, and library/API packaging remain future roadmap items.
- GitHub issue forms can be reconsidered later if Markdown templates prove insufficient.

</deferred>

---

*Phase: 08-release-and-maintenance-loop*
*Context gathered: 2026-06-06*

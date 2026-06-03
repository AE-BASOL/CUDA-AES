# Phase 1: Repository And Build Foundation - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 1 makes the repository clean, portable, and buildable by an outside developer. It does not fix AES-GCM correctness, add new AES modes, rewrite the benchmark harness, or perform SEO polish; those are later phases. It may add enough documentation and diagnostics to make build prerequisites clear.

</domain>

<decisions>
## Implementation Decisions

### Build Portability
- **D-01:** Remove maintainer-local absolute paths from `CMakeLists.txt` as part of Phase 1.
- **D-02:** Use CMake package discovery and cache variables rather than hard-coded CUDA/OpenSSL/Nsight paths.
- **D-03:** Support user-specified `CMAKE_CUDA_ARCHITECTURES`; do not hard-code only architecture `86` as the public default.
- **D-04:** CUDA host compiler setup must be documented or surfaced as an actionable configure-time diagnostic because the review failed when `nvcc` could not find `cl.exe`.

### Repository Hygiene
- **D-05:** Treat generated build output and IDE metadata as non-source. If any such files are tracked, Phase 1 should remove them from version control without touching unrelated source behavior.
- **D-06:** Keep top-level sources as the canonical implementation for now; `v3/` should be documented as duplicated/variant code or deferred for a later consolidation decision.

### Documentation Scope
- **D-07:** README changes in Phase 1 should focus on accurate build prerequisites and commands, not final SEO copy or performance marketing.
- **D-08:** Public docs must clearly state that GCM correctness findings are known blockers handled in Phase 2; Phase 1 should not imply current GCM output is standard AES-GCM.

### the agent's Discretion
- Exact CMake variable names, preset names, and diagnostic wording are left to the planner/implementer.
- Whether to add CMake presets in Phase 1 is flexible if they help make the build reproducible without over-expanding scope.

</decisions>

<specifics>
## Specific Ideas

- The imported review is the strongest concrete input for Phase 1: public configure failed because `nvcc` could not find `cl.exe`, and the CMake file still has private paths.
- The project should remain positioned as a reproducible benchmark suite, not a production cryptography library.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning
- `.planning/PROJECT.md` - project positioning, core value, constraints, and review-derived correctness gate.
- `.planning/REQUIREMENTS.md` - Phase 1 requirements `REPO-01`, `BUILD-01`, `BUILD-02`, `BUILD-03`, `BUILD-04`, and `BUILD-05`.
- `.planning/ROADMAP.md` - fixed Phase 1 boundary and success criteria.

### Codebase Context
- `.planning/codebase/STACK.md` - current CUDA/CMake/OpenSSL stack and hard-coded path details.
- `.planning/codebase/ARCHITECTURE.md` - current benchmark executable architecture.
- `.planning/codebase/CONCERNS.md` - build portability and imported review blockers.

### Imported Review
- `.planning/reviews/2026-06-04-main-branch-code-review.md` - main-branch review findings that drive Phase 1 build work and later correctness blockers.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CMakeLists.txt`: primary build definition to repair for public portability.
- `README.md`: current minimal build/run instructions; update only enough for accurate Phase 1 build guidance.
- `.gitignore`: already excludes common generated build artifacts and IDE files; verify tracked state separately before cleanup.

### Established Patterns
- Top-level source list is explicit in `CMakeLists.txt`.
- Current code builds one benchmark executable named `CudaProject`.
- OpenSSL is currently used as the CPU baseline through EVP in `main.cu`.

### Integration Points
- `CMakeLists.txt` should keep linking CUDA runtime and OpenSSL while replacing private path assumptions.
- Future Phase 2 correctness work depends on Phase 1 producing a build path that outside contributors can reproduce.

</code_context>

<deferred>
## Deferred Ideas

- Fix GCM tag verification, IV broadcast, and standard tag formula - Phase 2.
- Add NIST/OpenSSL correctness tests - Phase 2.
- Destroy CUDA events or introduce RAII for benchmark events - Phase 3.
- Full README SEO rewrite and GitHub metadata - Phase 4/7.
- Add CBC/CFB/OFB/CCM/XTS/KW/KWP modes - Phases 5 and 6.

</deferred>

---

*Phase: 01-repository-and-build-foundation*
*Context gathered: 2026-06-04*

# Phase 1: Repository And Build Foundation - Research

**Researched:** 2026-06-04
**Domain:** CUDA/CMake open-source build portability
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Remove maintainer-local absolute paths from `CMakeLists.txt` as part of Phase 1.
- D-02: Use CMake package discovery and cache variables rather than hard-coded CUDA/OpenSSL/Nsight paths.
- D-03: Support user-specified `CMAKE_CUDA_ARCHITECTURES`; do not hard-code only architecture `86` as the public default.
- D-04: CUDA host compiler setup must be documented or surfaced as an actionable configure-time diagnostic because the review failed when `nvcc` could not find `cl.exe`.
- D-05: Treat generated build output and IDE metadata as non-source.
- D-06: Keep top-level sources as the canonical implementation for now; `v3/` should be documented as duplicated/variant code or deferred for a later consolidation decision.
- D-07: README changes in Phase 1 should focus on accurate build prerequisites and commands, not final SEO copy or performance marketing.
- D-08: Public docs must clearly state that GCM correctness findings are known blockers handled in Phase 2; Phase 1 should not imply current GCM output is standard AES-GCM.

### the agent's Discretion
- Exact CMake variable names, preset names, and diagnostic wording.
- Whether to add CMake presets in Phase 1 if they reduce contributor friction without expanding scope.

### Deferred Ideas (OUT OF SCOPE)
- GCM correctness fixes, tag verification, IV broadcast, and standard tag formula.
- NIST/OpenSSL known-answer tests.
- CUDA event cleanup.
- SEO copy and GitHub metadata polish.
- New AES modes.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

Single-tier native CUDA/C++ benchmark application. Phase 1 owns repository/build infrastructure only:

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Portable CMake configuration | Build system | Documentation | CMake controls dependency discovery and compiler configuration. |
| Contributor build prerequisites | Documentation | Build system diagnostics | Users need both commands and actionable configure failures. |
| Repository hygiene | Version control | Documentation | Public source tree must distinguish canonical source from generated or variant files. |
</architectural_responsibility_map>

<research_summary>
## Summary

Phase 1 should repair the public build path without changing AES kernel behavior. The standard CMake approach is to use imported targets and cache variables: `find_package(CUDAToolkit REQUIRED)`, `find_package(OpenSSL REQUIRED)`, `target_link_libraries(... CUDA::cudart OpenSSL::SSL OpenSSL::Crypto)`, and user-configurable CUDA architectures.

CMake's CUDA host compiler variable must be set before CUDA is enabled by `project()`/`enable_language()`, typically through `-DCMAKE_CUDA_HOST_COMPILER=...` or a toolchain/preset, not as a placeholder at the bottom of `CMakeLists.txt`. The plan should remove that placeholder and document how Windows users launch from a Visual Studio Developer shell or pass the compiler explicitly.

**Primary recommendation:** Make top-level `CMakeLists.txt` portable first, then clean repository source boundaries, then update README build docs to match the new build path and known GCM blocker caveat.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Tool | Purpose | Why Standard |
|------|---------|--------------|
| CMake | Native build configuration | Existing project already uses it; supports CUDA language and imported dependency targets. |
| CUDAToolkit package | CUDA runtime target discovery | Provides `CUDA::cudart` target already used by the project. |
| FindOpenSSL | OpenSSL dependency discovery | Official CMake module provides imported targets instead of private paths. |
| CMake cache variables | User-configurable paths/options | Lets contributors pass architecture, optional profiler paths, or host compiler values without editing source. |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| CMakePresets.json | Repeatable configure presets | Useful if the executor can add simple cross-platform presets without overcomplicating Phase 1. |
| README build matrix | Human prerequisite documentation | Needed because CUDA host compiler setup differs on Windows/Linux. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `find_package(OpenSSL)` | Raw absolute `.lib` paths | Absolute paths block contributors. |
| `CMAKE_CUDA_ARCHITECTURES` cache variable | Hard-coded `86` only | Hard-code is convenient locally but wrong for open source. |
| Optional Nsight cache variable | Hard-coded `nsys.exe` path | Keeps profiling target available without breaking machines without Nsight. |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Build Configuration Flow

```
Developer command
  -> CMake configure
    -> CUDA language/toolkit discovery
    -> OpenSSL discovery
    -> optional Nsight path check
    -> target creation
      -> explicit CUDA source list
      -> CUDA::cudart + OpenSSL imported targets
  -> build CudaProject
```

### Recommended Project Structure For Phase 1

```
.
├── CMakeLists.txt              # portable top-level build
├── CMakePresets.json           # optional, if simple and useful
├── README.md                   # accurate build docs and known limitations
├── aes*.cu / aes_common.h      # canonical top-level implementation
├── v3/                         # documented variant, not canonical for Phase 1
└── cihangirTezcanAESimplementation/
```

### Pattern 1: Imported OpenSSL Targets
**What:** Use `find_package(OpenSSL REQUIRED)` and link `OpenSSL::SSL` / `OpenSSL::Crypto`.
**When to use:** Any CMake project linking OpenSSL.
**Why:** Imported targets carry include paths, link libraries, and platform-specific details.

### Pattern 2: User-Configurable CUDA Architecture
**What:** Let users pass `-DCMAKE_CUDA_ARCHITECTURES=86` or another architecture.
**When to use:** CUDA open-source projects where contributors have different GPUs.
**Why:** CMake initializes target `CUDA_ARCHITECTURES` from `CMAKE_CUDA_ARCHITECTURES` if set.

### Pattern 3: Host Compiler Configuration Before Project
**What:** Do not set `CMAKE_CUDA_HOST_COMPILER` after `project()`.
**When to use:** Windows CUDA builds needing a specific MSVC host compiler.
**Why:** CMake documents that the host compiler variable should be set before CUDA is enabled.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OpenSSL discovery | Custom include/lib path logic | `find_package(OpenSSL REQUIRED)` | Handles platform naming and imported targets. |
| CUDA toolkit linking | Manual CUDA include/lib paths | `find_package(CUDAToolkit REQUIRED)` + `CUDA::cudart` | Existing modern CMake path. |
| Architecture selection | Local `set(CMAKE_CUDA_ARCHITECTURES 86)` only | Cache/default with user override | Contributors use different GPUs. |
| Nsight path | Hard-coded maintainer executable | Optional cache variable + existence check | Profiling should not block builds. |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Setting CUDA host compiler too late
**What goes wrong:** CMake ignores or mishandles the value because CUDA was already enabled.
**How to avoid:** Remove bottom-of-file placeholder; document command-line/toolchain setup.
**Warning signs:** `nvcc cannot find cl.exe` during configure.

### Pitfall 2: Imported target not used
**What goes wrong:** Include paths or runtime libraries diverge across platforms.
**How to avoid:** Link imported targets, not raw `.lib` paths.
**Warning signs:** `libssl.lib` exists only on maintainer machine.

### Pitfall 3: Docs get ahead of correctness
**What goes wrong:** README implies GCM correctness before Phase 2 fixes.
**How to avoid:** Keep Phase 1 docs build-focused and include known limitation note.
**Warning signs:** README markets AES-GCM as standards-compliant benchmark output.
</common_pitfalls>

<open_questions>
## Open Questions

1. **Should CMakePresets.json be added now?**
   - What we know: Presets can simplify documented configure commands.
   - What's unclear: Whether the executor can create a useful preset without knowing every user environment.
   - Recommendation: Add only simple presets that do not encode private paths.

2. **Should `v3/` be moved now?**
   - What we know: Context says keep top-level sources canonical and document `v3/` as variant/deferred.
   - What's unclear: Whether `v3/` is tracked and intended public history.
   - Recommendation: Do not restructure `v3/` in Phase 1 unless it is tracked generated output; document its non-canonical status.
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- CMake FindOpenSSL docs: https://cmake.org/cmake/help/git-stage/module/FindOpenSSL.html
- CMake `CMAKE_CUDA_ARCHITECTURES` docs: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
- CMake `CMAKE_<LANG>_HOST_COMPILER` docs: https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_HOST_COMPILER.html
- Project review: `.planning/reviews/2026-06-04-main-branch-code-review.md`
- Codebase map: `.planning/codebase/STACK.md`
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: CMake, CUDA language, CUDAToolkit package, OpenSSL discovery.
- Patterns: imported targets, cache variables, host compiler diagnostics.
- Pitfalls: local paths, hard-coded architectures, overclaiming GCM correctness.

**Confidence breakdown:**
- Standard stack: HIGH - based on official CMake docs and current project stack.
- Architecture: HIGH - phase is narrow build infrastructure.
- Pitfalls: HIGH - review findings are concrete.
- Code examples: MEDIUM - exact implementation left to executor.

**Research date:** 2026-06-04
**Valid until:** 2026-07-04
</metadata>

---

*Phase: 01-repository-and-build-foundation*
*Research completed: 2026-06-04*
*Ready for planning: yes*

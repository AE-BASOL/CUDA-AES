# Phase 1: Repository And Build Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-06-04
**Phase:** 1-Repository And Build Foundation
**Areas discussed:** Imported review triage, build portability, repository hygiene, documentation scope

---

## Imported Review Triage

| Option | Description | Selected |
|--------|-------------|----------|
| Treat review as FYI only | Do not update planning artifacts. | |
| Fold review into planning | Store review and update requirements/roadmap/context. | Selected |
| Fix code immediately | Skip discussion and start implementation. | |

**User's choice:** User said the review should be incorporated, then Phase 1 should be discussed.
**Notes:** Review findings were added as planning constraints. GCM findings were assigned to Phase 2; CMake/build findings were assigned to Phase 1.

---

## Build Portability

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal README workaround | Tell users how to edit local paths manually. | |
| Portable CMake | Replace local paths with discovery/cache variables and diagnostics. | Selected |
| Full build-system rewrite | Restructure build from scratch. | |

**User's choice:** Inferred from review and project goals.
**Notes:** Portable CMake best matches open-source prestige and reproducibility.

---

## Repository Hygiene

| Option | Description | Selected |
|--------|-------------|----------|
| Leave generated files alone | Avoid cleanup in Phase 1. | |
| Clean tracked generated artifacts only | Remove generated/IDE files if tracked; keep source behavior stable. | Selected |
| Restructure entire source tree | Move all sources into new folders immediately. | |

**User's choice:** Inferred recommended default.
**Notes:** Large source restructuring would increase risk before build/correctness are stable.

---

## Documentation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Build docs only | Update build prerequisites and commands in Phase 1. | Selected |
| Full SEO README now | Do all public-facing polish before correctness work. | |
| No docs until release | Delay README updates. | |

**User's choice:** Inferred recommended default.
**Notes:** Final SEO copy should wait until correctness and benchmark methodology are credible.

## the agent's Discretion

- Exact CMake cache variable names.
- Whether to add CMake presets during Phase 1.
- Exact diagnostic wording for CUDA host compiler setup.

## Deferred Ideas

- GCM correctness fixes are Phase 2.
- CUDA event cleanup is Phase 3.
- SEO/discoverability is Phase 7.


# Phase 8: Release And Maintenance Loop - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md. This log preserves the alternatives considered.

**Date:** 2026-06-06
**Phase:** 8-release-and-maintenance-loop
**Areas discussed:** Release Package, Release Gate, Benchmark Contribution Checklist, Maintenance Loop

---

## Discussion Mode Note

The interactive question tool was unavailable in this runtime. Per the Codex adapter fallback for `AskUserQuestion`, the recommended scope was selected automatically and conservative defaults were used for the decisions.

---

## Release Package

| Option | Description | Selected |
|--------|-------------|----------|
| GitHub Release plus verified artifacts | Use `v1.0.0`, release notes, changelog promotion, and attach only verified benchmark artifacts. | Yes |
| Release notes only | Publish tag and notes without benchmark artifacts. | |
| Artifact bundle first | Focus mainly on packaging raw benchmark files and generated summary. | |

**User's choice:** Fallback selected the recommended conservative option.
**Notes:** Existing `CHANGELOG.md`, `docs/results.md`, and `docs/benchmark-methodology.md` already define most release-note and artifact expectations.

---

## Release Gate

| Option | Description | Selected |
|--------|-------------|----------|
| Gate final v1 on runtime verification | Require Release build, CTest, smoke benchmark, and summary generation in a CUDA host compiler environment. | Yes |
| Ship with known verification debt | Publish final v1 even if current shell cannot run CUDA verification. | |
| Prepare release candidate only | Produce all notes/checklists but leave publication to maintainer. | |

**User's choice:** Fallback selected the recommended conservative option.
**Notes:** The project core value depends on reproducibility and correctness evidence. If the environment remains blocked, planning should produce release-candidate materials rather than a final release claim.

---

## Benchmark Contribution Checklist

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated canonical checklist | Add one checklist doc and link it from contributing docs, PR template, issue template, and results docs. | Yes |
| Keep checklist duplicated | Leave detailed requirements spread across existing files. | |
| Convert to GitHub issue forms | Replace Markdown templates with structured issue forms. | |

**User's choice:** Fallback selected the recommended conservative option.
**Notes:** Existing Markdown templates are sufficient for v1. A canonical checklist reduces drift while preserving the current lightweight repository style.

---

## Maintenance Loop

| Option | Description | Selected |
|--------|-------------|----------|
| Compact documented loop | Define changelog, roadmap, issue triage, benchmark review, and security reporting expectations. | Yes |
| Minimal changelog only | Only update `CHANGELOG.md` and leave process implicit. | |
| Heavy governance process | Add more formal roles, SLAs, and policy machinery. | |

**User's choice:** Fallback selected the recommended conservative option.
**Notes:** The repo should stay lightweight but concrete. Security reporting needs clearer guidance than the current "private maintainer channel if available" wording.

---

## the agent's Discretion

- Exact filename for the benchmark checklist.
- Exact release note wording and artifact bundle naming.
- Whether the maintenance loop is a new doc or an existing governance section.

## Deferred Ideas

- GitHub Pages or standalone project website.
- DOI-backed archived release and paper-style publication.
- GMAC/CMAC benchmarking, charts, benchmark matrix automation, and library/API packaging.
- GitHub issue forms.

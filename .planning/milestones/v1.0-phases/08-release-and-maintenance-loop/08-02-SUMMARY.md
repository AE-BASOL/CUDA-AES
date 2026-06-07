---
phase: 08-release-and-maintenance-loop
plan: 02
subsystem: governance
tags: [maintenance, security-policy, issue-templates, github]
requires:
  - phase: 04-open-source-documentation-package
    provides: "Baseline governance files and GitHub Markdown templates"
provides:
  - "Private-reporting-first security policy with safe public fallback"
  - "Post-v1 maintenance loop covering triage, changelog, roadmap, benchmark review, security, and v2 boundaries"
  - "Bug and AES mode request templates with GitHub chooser frontmatter"
affects: [release-and-maintenance-loop, contributor-guidance, security-reporting, roadmap-maintenance]
tech-stack:
  added: []
  patterns: [GitHub Markdown issue templates, maintenance-loop documentation]
key-files:
  created:
    - docs/maintenance.md
    - .planning/phases/08-release-and-maintenance-loop/08-USER-SETUP.md
  modified:
    - SECURITY.md
    - README.md
    - .github/ISSUE_TEMPLATE/bug_report.md
    - .github/ISSUE_TEMPLATE/mode_request.md
key-decisions:
  - "Preferred GitHub private vulnerability reporting when enabled while preserving a public no-sensitive-details fallback."
  - "Kept v2 work separate from v1 maintenance unless the roadmap changes."
  - "Preserved Markdown issue templates for v1 rather than moving to GitHub issue forms."
patterns-established:
  - "Maintenance docs should distinguish v1 upkeep from future v2 expansion work."
  - "Security reporting copy should avoid asking for sensitive details in public issues."
requirements-completed: [MAINT-01, MAINT-03, MAINT-04]
requirements-pending: []
duration: 5 min
completed: 2026-06-06
---

# Phase 8 Plan 02: Security And Maintenance Loop Summary

**Release-ready security reporting, post-v1 maintenance loop, and useful bug/mode request templates**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-06T21:14:10+03:00
- **Completed:** 2026-06-06T21:19:06+03:00
- **Tasks:** 3 completed
- **Files modified:** 6 tracked deliverable files

## Accomplishments

- Updated `SECURITY.md` with GitHub private vulnerability reporting as the preferred path when enabled and a public fallback that avoids sensitive details.
- Added `docs/maintenance.md` with triage categories, changelog upkeep, roadmap triggers, benchmark-result review, security handling, and future-v2 boundaries.
- Linked maintenance documentation from the README governance area.
- Added GitHub Markdown-template frontmatter and more precise prompts to bug report and AES mode request templates.
- Created `08-USER-SETUP.md` for the maintainer-only GitHub private vulnerability reporting setting.

## Task Commits

1. **Task 1: Make SECURITY.md release-ready** - `b8a2e63` (`docs(08-02): clarify security reporting path`)
2. **Task 2: Add maintenance loop documentation** - `429fbaf` (`docs(08-02): add maintenance loop`)
3. **Task 3: Tighten bug and mode request templates** - `29cafbc` (`docs(08-02): tighten issue templates`)

**Plan metadata:** committed after summary creation.

## Files Created/Modified

- `SECURITY.md` - Private-reporting-first guidance, safe public fallback, and benchmark/research scope.
- `docs/maintenance.md` - Post-v1 maintenance loop and v2 boundary.
- `README.md` - Governance link to maintenance docs.
- `.github/ISSUE_TEMPLATE/bug_report.md` - Template chooser frontmatter and clearer command/artifact prompts.
- `.github/ISSUE_TEMPLATE/mode_request.md` - Template chooser frontmatter and benchmark/research scope prompt.
- `.planning/phases/08-release-and-maintenance-loop/08-USER-SETUP.md` - Maintainer setup checklist for GitHub private vulnerability reporting.

## Decisions Made

- Kept the security policy concrete without implying production cryptography library support.
- Treated private vulnerability reporting enablement as a maintainer-only GitHub settings action.
- Kept GMAC/CMAC benchmarking, charts, matrix automation, GitHub Pages, DOI releases, reports, and library/API packaging in future-v2 scope.

## Deviations from Plan

None - plan executed exactly as written.

**Total deviations:** 0 auto-fixed.
**Impact on plan:** No scope changes.

## Issues Encountered

None.

## User Setup Required

External repository settings require manual configuration. See [08-USER-SETUP.md](./08-USER-SETUP.md) for:

- GitHub private vulnerability reporting setting location.
- When to enable it.
- Verification note for keeping `SECURITY.md` fallback guidance accurate.

## Verification

- `rg "private vulnerability reporting|not include sensitive|preferred security contact|benchmark and research software|not a production cryptography library" SECURITY.md` - passed.
- `Test-Path docs\maintenance.md` - passed.
- `rg "build|correctness|benchmark result|documentation|security|mode request|CHANGELOG|Unreleased|roadmap|v2" docs\maintenance.md` - passed.
- `rg "docs/maintenance.md|Maintenance" README.md` - passed.
- `rg "name: Bug Report|about:|Environment|Commands|Artifacts" .github\ISSUE_TEMPLATE\bug_report.md` - passed.
- `rg "name: AES Mode Request|about:|Standards Reference|Expected Coverage|benchmark/research" .github\ISSUE_TEMPLATE\mode_request.md` - passed.

## Self-Check: PASSED

All source-controlled task acceptance criteria and plan-level verification commands passed. The repository now has a concrete maintenance loop, safe security reporting path, and more useful maintenance templates.

## Next Phase Readiness

Plan 08-02 is complete. Wave 2 can prepare the v1.0.0 release package using the finalized contribution, security, and maintenance surfaces.

---
*Phase: 08-release-and-maintenance-loop*
*Completed: 2026-06-06*

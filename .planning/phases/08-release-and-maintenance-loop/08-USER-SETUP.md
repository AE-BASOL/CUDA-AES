# Phase 8: User Setup Required

**Generated:** 2026-06-06
**Phase:** 08-release-and-maintenance-loop
**Status:** Incomplete

Complete these items if the repository maintainer wants GitHub-native private vulnerability reporting. The source-controlled security policy now documents this path, but enabling the repository setting requires maintainer admin access in GitHub.

## Environment Variables

None.

## Account Setup

None.

## Dashboard Configuration

- [ ] **Enable GitHub private vulnerability reporting**
  - Location: GitHub repository -> Settings -> Code security and analysis -> Private vulnerability reporting
  - Set to: Enabled, if the maintainer wants security researchers to report privately through GitHub
  - Skip if: The repository owner prefers a different private maintainer contact path

## Verification

After completing setup, verify in GitHub repository settings that private vulnerability reporting is enabled. If it is not enabled, keep `SECURITY.md` as the source of truth for the public fallback: reporters should ask for the preferred security contact without disclosing sensitive details.

---

**Once all items complete:** Mark status as "Complete" at top of file.

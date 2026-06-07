# Phase 7: Discoverability And SEO - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md - this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 07-discoverability-and-seo
**Areas discussed:** Search Metadata, README Terms, Docs Structure, GitHub Pages

---

## Search Metadata

| Option | Description | Selected |
|--------|-------------|----------|
| Focused benchmark metadata | Description and topics center on `cuda-aes`, `gpu-aes`, `aes-benchmark`, `cuda-benchmark`, `cryptography-benchmark`, `reproducible-benchmarks`. | Yes |
| Broad cryptography metadata | Include wider terms like `encryption`, `gpu-cryptography`, `aes-gcm`, `aes-xts`, `aes-ccm`. | |
| Maximum mode coverage metadata | Emphasize every mode in topics/docs metadata where possible. | |

**User's choice:** Focused benchmark metadata.
**Notes:** This keeps the repository identity aligned with reproducible CUDA/GPU AES benchmarking rather than broad cryptography marketing.

---

## README Terms

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal natural rewrite | Keep the current first paragraph mostly intact and add target terms naturally in headings/subheadings. | |
| SEO-forward intro | Rewrite the opening paragraph to include more target phrases directly. | Yes |
| Technical-reader first | Optimize primarily for trust signals such as coverage, correctness, artifacts, and limitations. | |

**User's choice:** SEO-forward intro.
**Notes:** The rewrite should still read as technical documentation and keep the benchmark/research software disclaimer near the top.

---

## Docs Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Retitle existing pages only | Keep current filenames and improve H1s/link labels. | |
| Add a discoverability index | Keep existing pages and add one new docs landing page focused on search-friendly navigation. | Yes |
| Rename files for keywords | Change filenames to keyword-rich names, accepting link churn or redirects. | |

**User's choice:** Add a discoverability index.
**Notes:** Existing filenames should stay stable. The new page should improve navigation across modes, benchmarks, correctness, results, and related docs.

---

## GitHub Pages

| Option | Description | Selected |
|--------|-------------|----------|
| Prepare optional Pages plan only | Add docs/content structure that could later become GitHub Pages, without adding a generator or workflow. | |
| Add simple GitHub Pages now | Create a minimal static landing page or docs publishing setup in this phase. | |
| Skip Pages entirely | Keep all discoverability work inside GitHub README/docs/repo metadata. | Yes |

**User's choice:** Skip Pages entirely.
**Notes:** GitHub Pages is out of scope for Phase 7 and can be reconsidered later.

---

## the agent's Discretion

- Exact README wording, final topic list formatting, and docs landing-page filename are left to downstream planning and implementation.

## Deferred Ideas

- GitHub Pages or a standalone static project site can be reconsidered later.
- Versioned releases with reproducibility notes remain Phase 8.

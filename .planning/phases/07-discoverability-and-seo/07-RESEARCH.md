# Phase 7: Discoverability And SEO - Research

**Researched:** 2026-06-05
**Status:** Complete

## Research Question

What does the executor need to know to plan discoverability improvements for a CUDA AES benchmark repository without weakening the project's technical credibility?

## Phase Scope

Phase 7 targets DOCS-04: search engines and GitHub search should infer that the repository is about CUDA AES, GPU AES benchmark, AES GPU performance, and reproducible cryptography benchmarking.

This is a documentation and metadata phase. It should not change CUDA source, tests, benchmark behavior, release packaging, or GitHub Pages.

## Current State

- `README.md` already has a credible first screen and top-level disclaimer.
- Existing docs cover architecture, correctness, benchmark methodology, results, profiling, mode matrix, and legacy provenance.
- `CITATION.cff` already provides project metadata and can be used as a consistency source for title/abstract/keywords.
- `.planning/phases/07-discoverability-and-seo/07-CONTEXT.md` locks the core decisions:
  - focused benchmark metadata, not cryptography marketing;
  - SEO-forward README terms without keyword stuffing;
  - new docs landing page with stable existing doc filenames;
  - no GitHub Pages work in this phase.

## Implementation Implications

### GitHub Metadata

GitHub repository metadata is partly outside the source tree. The plan should produce exact maintainer-facing values for:

- repository description;
- repository topics.

Recommended description shape:

> Reproducible CUDA AES benchmark suite for GPU AES modes, correctness checks, and raw benchmark artifacts.

Recommended topics:

- `cuda-aes`
- `gpu-aes`
- `aes-benchmark`
- `cuda-benchmark`
- `cryptography-benchmark`
- `reproducible-benchmarks`
- `aes-gcm`
- `aes-ctr`
- `gpu-cryptography`

The executor should store these recommendations in a source-controlled doc or README section because setting repository metadata itself requires GitHub UI/API access.

### README

The README already opens with "CUDA-AES Benchmark" and "reproducible GPU AES benchmark suite." Phase 7 should refine rather than replace it.

Useful terms to naturally include near the first screen:

- CUDA AES benchmark
- GPU AES benchmark
- AES GPU performance
- AES-GCM CUDA
- AES-128 CUDA
- AES-256 CUDA
- reproducible cryptography benchmark

Risk: repeating these as a keyword block would damage technical quality. Use them in headings, a concise "Search and scope" paragraph, and documentation link labels.

### Docs Landing Page

A `docs/README.md` landing page is the lowest-risk option:

- It does not rename existing files.
- It gives GitHub a natural docs index.
- It can use descriptive link text for all existing pages.
- It can link back to the top-level README.

The landing page should route readers to:

- mode coverage;
- benchmark methodology;
- correctness;
- results;
- architecture;
- profiling;
- legacy/provenance.

### GitHub Pages

`07-CONTEXT.md` explicitly says not to prepare or add GitHub Pages. The roadmap success criterion that mentions optional GitHub Pages should be handled by documenting the deferral: Phase 7 keeps docs GitHub-native; standalone Pages remains future work if README/docs indexing is insufficient.

## Verification Strategy

Check for natural discoverability language and guardrails:

- `rg "CUDA AES benchmark|GPU AES benchmark|AES GPU performance|AES-GCM CUDA|reproducible cryptography benchmark" README.md docs CITATION.cff`
- `rg "not a production cryptography library|does not claim|unsupported|fastest" README.md docs`
- `Test-Path docs/README.md`
- `rg "Mode Matrix|Benchmark Methodology|Correctness|Results|Architecture|Profiling|Legacy" docs/README.md`
- `rg "GitHub Pages|future|deferred" README.md docs .planning/phases/07-discoverability-and-seo`

## Planning Recommendation

Use one autonomous execution plan in wave 1. The work is cohesive, low-risk, and documentation-only. The plan should modify:

- `README.md`
- `docs/README.md`
- possibly `CITATION.cff` if keyword consistency is weak
- `.planning/phases/07-discoverability-and-seo/07-VERIFICATION.md`

The plan must include a threat model for overclaiming, keyword stuffing, and accidental production-library positioning.

## RESEARCH COMPLETE

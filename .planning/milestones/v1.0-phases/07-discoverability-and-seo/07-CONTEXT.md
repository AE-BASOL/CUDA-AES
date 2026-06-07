# Phase 7: Discoverability And SEO - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 7 improves GitHub search, web discoverability, and technical-reader entry points for the CUDA-AES Benchmark repository. It should optimize repository metadata, README wording, and docs navigation for DOCS-04 without adding unsupported performance claims, keyword stuffing, a production cryptography-library positioning, or a GitHub Pages site.

</domain>

<decisions>
## Implementation Decisions

### Search Metadata
- **D-01:** Use focused benchmark metadata rather than broad cryptography marketing.
- **D-02:** GitHub description and topics should center on terms such as `cuda-aes`, `gpu-aes`, `aes-benchmark`, `cuda-benchmark`, `cryptography-benchmark`, and `reproducible-benchmarks`.
- **D-03:** Avoid generic or hype-oriented metadata that implies a production encryption library or unsupported performance leadership.

### README Terms
- **D-04:** Use an SEO-forward README introduction that includes target phrases more directly than the current copy.
- **D-05:** The SEO-forward rewrite must still read like technical documentation, not a keyword block.
- **D-06:** Keep the benchmark/research software disclaimer near the top of the README so discoverability improvements do not weaken the project's credibility guardrails.

### Docs Structure
- **D-07:** Add a search-friendly docs landing page for CUDA AES benchmark navigation.
- **D-08:** Keep existing documentation filenames stable; do not rename current docs for keyword-only URL gains.
- **D-09:** The new landing page should route readers to CUDA AES modes, benchmark methodology, correctness, results, architecture, profiling, and legacy/provenance material using descriptive link text.

### GitHub Pages
- **D-10:** Do not prepare or add GitHub Pages in Phase 7.
- **D-11:** Keep this phase scoped to GitHub README/docs/repository metadata.

### the agent's Discretion
- Exact README phrasing, metadata topic list finalization, and docs landing-page title are left to the planner/implementer, constrained by the decisions above.
- The planner may choose whether the docs landing page is linked as `docs/README.md`, `docs/index.md`, or another stable filename if it preserves existing links and improves reader navigation.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Planning
- `.planning/PROJECT.md` - project positioning, SEO target terms, credibility constraints, and current Phase 7 context.
- `.planning/REQUIREMENTS.md` - DOCS-04 requirement and traceability for Phase 7.
- `.planning/ROADMAP.md` - fixed Phase 7 boundary and success criteria.
- `.planning/STATE.md` - current project state and carried verification debt.

### Prior Decisions
- `.planning/phases/01-repository-and-build-foundation/01-CONTEXT.md` - prior decisions preserving reproducible benchmark positioning and deferring final SEO copy to later phases.

### Codebase Context
- `.planning/codebase/CONVENTIONS.md` - documentation conventions and current README/doc style.
- `.planning/codebase/STRUCTURE.md` - repository layout, docs folder, and canonical source boundary.
- `.planning/codebase/STACK.md` - CUDA/CMake/OpenSSL stack and public project identity.

### Public Documentation
- `README.md` - primary public landing page to tune for first-screen SEO and credibility.
- `docs/benchmark-methodology.md` - reproducible benchmark methodology page.
- `docs/correctness.md` - correctness gate and verification limits.
- `docs/results.md` - result packaging and claim constraints.
- `docs/modes.md` - AES mode matrix and mode-specific scope.
- `docs/architecture.md` - current benchmark architecture.
- `docs/profiling.md` - profiling guidance.
- `docs/legacy-tezcan.md` - provenance and legacy-claim boundaries.
- `CITATION.cff` - citation metadata and existing keyword signals.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `README.md`: already contains a credible benchmark-suite intro, coverage table, quick start, docs hub, limitations, and roadmap direction; Phase 7 should refine rather than replace wholesale.
- `docs/`: existing documentation pages cover architecture, benchmark methodology, correctness, results, profiling, mode matrix, and legacy provenance.
- `CITATION.cff`: already includes title, abstract, and keywords that can inform consistent metadata wording.

### Established Patterns
- Public docs use simple Markdown files with direct relative links from the README.
- Current docs emphasize reproducibility, correctness gates, timing-scope distinctions, and limitations before claims.
- Existing file names are clear and already linked from README, so link stability matters.

### Integration Points
- GitHub repository metadata must be handled outside the normal source tree, but the planner should document the exact description/topics to apply.
- README changes should preserve the first-screen identity and disclaimer.
- A new docs landing page should be linked from README and should link back to existing docs without renaming them.

</code_context>

<specifics>
## Specific Ideas

- Preferred metadata direction: focused benchmark terms such as `cuda-aes`, `gpu-aes`, `aes-benchmark`, `cuda-benchmark`, `cryptography-benchmark`, and `reproducible-benchmarks`.
- README should be SEO-forward, but still sound like a serious technical project.
- Add a docs landing page rather than renaming existing files.
- Skip GitHub Pages entirely for Phase 7.

</specifics>

<deferred>
## Deferred Ideas

- GitHub Pages or a standalone static project site can be reconsidered in a later phase if GitHub README/docs discoverability is not enough.
- Versioned releases with reproducibility notes remain Phase 8.

</deferred>

---

*Phase: 07-discoverability-and-seo*
*Context gathered: 2026-06-05*

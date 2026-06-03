# Research: Pitfalls

## Pitfall: Public Benchmark Before Correctness

Warning signs:
- Throughput tables exist before known-answer tests.
- GCM tag behavior is unclear.
- CPU baseline is used only for speed, not correctness.

Prevention:
- Add AES vector tests first.
- Document exactly what modes and semantics are verified.
- Mark benchmark results as preliminary until tests pass in CI/manual verification.

## Pitfall: Non-Reproducible Build

Warning signs:
- Absolute paths in CMake.
- README commands do not match actual platform.
- Required CUDA/OpenSSL versions are not stated.

Prevention:
- Use portable CMake discovery.
- Document supported platforms.
- Add a clean build script and CMake presets.

## Pitfall: SEO Without Substance

Warning signs:
- README repeats keywords but lacks methodology.
- No docs pages for benchmark methodology, results, and correctness.
- No GitHub topics or project description.

Prevention:
- Use natural headings that match real user intent.
- Publish useful docs that developers would bookmark.
- Add GitHub Pages only after core docs are accurate.

## Pitfall: Ambiguous Open-Source Governance

Warning signs:
- No license.
- No contribution policy.
- No vulnerability reporting guidance.
- No citation metadata.

Prevention:
- Add `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, `CHANGELOG.md`, and issue/PR templates.

## Sources

- Google SEO Starter Guide: https://developers.google.cn/search/docs/fundamentals/seo-starter-guide?hl=en
- GitHub repository best practices: https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories
- Johns Hopkins OSPO public code repository best practices: https://ospo.library.jhu.edu/learn-grow/public-code-repository-best-practices/


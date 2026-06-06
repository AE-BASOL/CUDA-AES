# Pull Request

## Summary

What changed?

## Scope

- [ ] Build portability
- [ ] Correctness tests
- [ ] Benchmark harness
- [ ] AES mode implementation
- [ ] Documentation
- [ ] Governance / maintenance

## Verification

Paste relevant commands and results:

```text
ctest --test-dir build --output-on-failure
```

If benchmark behavior changed, include:

- benchmark command
- `run_metadata.csv`
- `thr_gpu.csv`
- `thr_cpu.csv`
- `summary.md`

Benchmark-changing PRs should follow `docs/benchmark-result-contributions.md`:
[benchmark result contribution checklist](../docs/benchmark-result-contributions.md).

## Security And Scope

- [ ] This does not imply production cryptography library safety.
- [ ] Cryptographic behavior changes include deterministic tests.
- [ ] Benchmark claims are backed by raw artifacts.

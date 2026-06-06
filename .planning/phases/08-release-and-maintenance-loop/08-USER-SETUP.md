# Phase 8: User Setup Required

**Generated:** 2026-06-06
**Phase:** 08-release-and-maintenance-loop
**Status:** Incomplete

Complete these items before treating v1 as fully published. The source-controlled docs and release-candidate artifacts are prepared, but these items require a CUDA/MSVC-ready shell or maintainer access to GitHub repository settings.

## Environment Variables

None.

## Account Setup

None.

## Dashboard Configuration

- [ ] **Enable GitHub private vulnerability reporting**
  - Location: GitHub repository -> Settings -> Code security and analysis -> Private vulnerability reporting
  - Set to: Enabled, if the maintainer wants security researchers to report privately through GitHub
  - Skip if: The repository owner prefers a different private maintainer contact path

- [ ] **Publish or save the GitHub Release draft**
  - Location: GitHub repository -> Releases -> Draft a new release
  - Tag: `v1.0.0`
  - Title: `CUDA-AES Benchmark v1.0.0`
  - Notes source: `docs/release-v1.md`
  - Required before publish: runtime release gate has `verification-passed`
  - Assets: attach only raw artifacts from the verified release-gate run

## Local Release Gate

- [ ] **Run the final release gate from a CUDA/MSVC-ready shell**
  - Use: Visual Studio Developer Command Prompt, or pass `-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>`
  - Commands:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
cmake --build build --config Release
ctest --test-dir build --output-on-failure
.\build\Release\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke
python scripts\summarize_benchmarks.py bench\v1-smoke\thr_gpu.csv bench\v1-smoke\thr_cpu.csv -o bench\v1-smoke\summary.md
```

  - If the executable is produced at `.\build\CudaProject.exe`, use that path instead.
  - After a pass, update `08-VERIFICATION.md`, `docs/release-v1.md`, and `CHANGELOG.md` from release candidate to final publication state.

## Verification

After completing setup:

- verify in GitHub repository settings that private vulnerability reporting is enabled, if that path was selected;
- verify `.planning/phases/08-release-and-maintenance-loop/08-VERIFICATION.md` records `verification-passed`;
- verify the GitHub Release draft uses tag `v1.0.0` and only includes artifacts from the verified release-gate run.

---

**Once all items complete:** Mark status as "Complete" at top of file.

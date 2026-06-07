# Phase 8: User Setup Required

**Generated:** 2026-06-06
**Phase:** 08-release-and-maintenance-loop
**Status:** Published — only the optional private vulnerability reporting toggle remains a maintainer choice.

The local CUDA/MSVC release gate passed and the `v1.0.0` GitHub Release is published (2026-06-06). The single remaining item below is the optional GitHub private vulnerability reporting setting, which is the maintainer's discretion.

## Environment Variables

None.

## Account Setup

None.

## Dashboard Configuration

- [ ] **Enable GitHub private vulnerability reporting**
  - Location: GitHub repository -> Settings -> Code security and analysis -> Private vulnerability reporting
  - Set to: Enabled, if the maintainer wants security researchers to report privately through GitHub
  - Skip if: The repository owner prefers a different private maintainer contact path

- [x] **Publish or save the GitHub Release draft**
  - Published: 2026-06-06 via `gh release create v1.0.0` (full release, not draft).
  - URL: https://github.com/AE-BASOL/CUDA-AES/releases/tag/v1.0.0
  - Tag: `v1.0.0` (annotated, points at `d63bff9`)
  - Title: `CUDA-AES Benchmark v1.0.0`
  - Notes source: `docs/release-v1.md`
  - Assets attached: `run_metadata.csv`, `thr_gpu.csv`, `thr_cpu.csv`, `summary.md` from `bench\v1-smoke-local`

## Local Release Gate

- [x] **Run the final release gate from a CUDA/MSVC-ready shell**
  - Completed: 2026-06-06 from a Visual Studio 2022 Developer Command Prompt.
  - Build dir: `build-vs2022-release3`
  - Artifact dir: `bench\v1-smoke-local`
  - Commands run:

```powershell
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\cmake.exe" --build build-vs2022-release3 --config Release'
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && "C:\Program Files\CMake\bin\ctest.exe" --test-dir build-vs2022-release3 --output-on-failure'
cmd.exe /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && build-vs2022-release3\CudaProject.exe --runs 1 --sizes 1048576 --bench-dir bench\v1-smoke-local'
python scripts\summarize_benchmarks.py bench\v1-smoke-local\thr_gpu.csv bench\v1-smoke-local\thr_cpu.csv -o bench\v1-smoke-local\summary.md
```

## Verification

After completing setup:

- verify in GitHub repository settings that private vulnerability reporting is enabled, if that path was selected;
- verify `.planning/phases/08-release-and-maintenance-loop/08-VERIFICATION.md` records `verification-passed`;
- verify the GitHub Release draft uses tag `v1.0.0` and only includes artifacts from the verified release-gate run.

---

**Once GitHub publication items complete:** Mark status as "Complete" at top of file.

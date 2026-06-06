# Phase 8: User Setup Required

**Generated:** 2026-06-06
**Phase:** 08-release-and-maintenance-loop
**Status:** Pending external publication

Complete these items before treating v1 as fully published on GitHub. The local CUDA/MSVC release gate has passed; remaining items require maintainer access to GitHub repository settings or release publication.

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
  - Required before publish: runtime release gate has `verification-passed` in `08-VERIFICATION.md`
  - Assets: attach only raw artifacts from `bench\v1-smoke-local`

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

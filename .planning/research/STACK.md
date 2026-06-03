# Research: Stack

## Question

What technical stack and repository assets are needed for a credible open-source CUDA AES benchmark suite?

## Findings

- Keep CUDA C++ and CMake as the primary build path because the existing code is already organized around `CMakeLists.txt`, `.cu` files, and `CUDA::cudart`.
- Replace hard-coded local paths with portable CMake discovery:
  - `find_package(CUDAToolkit REQUIRED)`
  - `find_package(OpenSSL REQUIRED)`
  - cache variables for optional Nsight/NVTX settings
  - user-selectable `CMAKE_CUDA_ARCHITECTURES`
- Add CTest-compatible correctness tests so contributors can run `ctest` after build.
- Keep OpenSSL as the CPU baseline, but check EVP return values and document exact OpenSSL version in benchmark reports.
- Add scripts under `scripts/` for repeatable benchmark runs and environment capture.
- Add docs under `docs/` for methodology, hardware setup, correctness, and results.

## Recommended Repository Assets

- `README.md` as the main GitHub landing page.
- `LICENSE` for legal clarity.
- `CONTRIBUTING.md` for build, test, benchmark, and PR expectations.
- `SECURITY.md` because cryptography-related repos need explicit vulnerability reporting boundaries.
- `CITATION.cff` so researchers and blog authors can cite the software.
- `CHANGELOG.md` for release history.
- `.github/ISSUE_TEMPLATE/` and `.github/pull_request_template.md`.
- `docs/benchmark-methodology.md`, `docs/results.md`, and `docs/correctness.md`.

## Sources

- GitHub repository best practices: https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories
- Johns Hopkins OSPO public code repository best practices: https://ospo.library.jhu.edu/learn-grow/public-code-repository-best-practices/


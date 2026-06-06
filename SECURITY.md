# Security Policy

CUDA-AES Benchmark is benchmark and research software. It is not a production cryptography library and should not be used as a drop-in security component.

## Supported Scope

Security-sensitive reports are welcome when they affect:

- correctness of benchmarked AES outputs
- authentication/tag behavior in the supported GCM scope
- misleading documentation that could imply production safety
- build or dependency issues that affect users running the benchmark

The current supported GCM scope is 96-bit IV, empty AAD, and full 16-byte blocks.

## Out of Scope

- Requests for production cryptography API guarantees
- Vulnerabilities in future modes that are not implemented yet
- Performance-only issues without correctness or reproducibility impact
- Local benchmark results that are not accompanied by raw artifacts and environment details

## Reporting

Use GitHub private vulnerability reporting when it is enabled for this repository. That path lets maintainers review security-sensitive reports before any public disclosure.

If private vulnerability reporting is not enabled and no preferred security contact is listed by the maintainer, open a public GitHub issue asking for the preferred security contact. Do not include sensitive details, private keys, secrets, exploit payloads, or sensitive production data in that public issue.

For non-sensitive issues, open a regular GitHub issue and include:

- affected commit
- build environment
- CUDA Toolkit and driver version
- GPU model
- correctness or benchmark command
- expected vs actual behavior

Security reports should describe the benchmark/research impact clearly and should not request production cryptography library guarantees for v1.

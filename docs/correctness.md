# Correctness

Correctness checks are the gate before interpreting benchmark output.

## CTest

Build the project, then run:

```bash
ctest --test-dir build --output-on-failure
```

The CTest suite runs `CudaAesKat`, implemented in `tests/kat_main.cu`.

## Known-Answer Coverage

Current deterministic known-answer tests cover:

- AES-128 ECB encrypt/decrypt
- AES-256 ECB encrypt/decrypt
- AES-128 CTR encrypt/decrypt
- AES-256 CTR encrypt/decrypt
- AES-128 CBC encrypt/decrypt
- AES-256 CBC encrypt/decrypt
- AES-128 CFB-128 encrypt/decrypt
- AES-256 CFB-128 encrypt/decrypt
- AES-128 OFB encrypt/decrypt
- AES-256 OFB encrypt/decrypt
- AES-128 GCM ciphertext/tag/decrypt tag
- AES-256 GCM ciphertext/tag/decrypt tag
- GCM wrong-tag rejection
- GCM tampered-ciphertext rejection
- AES-128 CCM ciphertext/tag/decrypt tag
- AES-256 CCM ciphertext/tag/decrypt tag
- CCM wrong-tag rejection
- AES-128-XTS full-block encrypt/decrypt
- AES-256-XTS full-block encrypt/decrypt

## GCM Scope

Current GCM correctness scope is:

- 96-bit IV
- empty AAD
- full 16-byte blocks
- ciphertext/tag verification
- wrong-tag and tampered-ciphertext rejection

Non-empty AAD, partial-block behavior, and a production AEAD API are future work.

## CCM Scope

Current CCM correctness scope is:

- 96-bit nonce
- empty AAD
- 16-byte authentication tag
- full 16-byte blocks
- ciphertext/tag verification
- wrong-tag rejection

Non-empty AAD, partial-block behavior, variable nonce lengths, variable tag lengths, and a production AEAD API are future work.

## XTS-AES Scope

Current XTS-AES correctness scope is:

- AES-128-XTS with 256 bits of total key material
- AES-256-XTS with 512 bits of total key material
- 16-byte sector tweak input
- full 16-byte blocks
- encrypt/decrypt known-answer checks

Ciphertext stealing for non-block-multiple data units is not implemented in the current benchmark scope.

## Confidentiality Mode Scope

ECB, CBC, CFB, OFB, and CTR are confidentiality-only modes. They do not provide authentication. Phase 5 CFB coverage uses CFB-128 full-block segment semantics only; smaller CFB segment sizes are not part of the current KAT or benchmark scope.

## Environment Limitation

In the current shell, runtime CMake/CTest verification is blocked because `nvcc` cannot find the Visual Studio host compiler `cl.exe`. Run from a Visual Studio Developer Command Prompt or pass:

```powershell
-DCMAKE_CUDA_HOST_COMPILER=<path-to-cl.exe>
```

This is tracked as environment-limited verification debt, not a source-level correctness failure.

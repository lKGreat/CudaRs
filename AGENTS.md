# Repository Guidelines

## Project Structure & Module Organization
- `build-support/cuda-build/` houses shared build tooling for CUDA detection/linking.
- `cuda-sys/*` contains raw FFI crates (`*-sys`) for CUDA libraries.
- `cuda-rs/*` contains safe Rust wrappers that layer on top of the FFI crates.
- `cudars-ffi/` builds the native `cdylib` consumed by the .NET packages.
- `dotnet/` holds the C# solution: `CudaRS.Native`, `CudaRS.Core`, `CudaRS`, `CudaRS.Examples`, and `CudaRS.Yolo`.

## Build, Test, and Development Commands
- `cargo build --release` builds Rust crates with the default CUDA 12 feature set.
- `cargo build --release --features cuda-11` or `cargo build --release --features cuda-12-3` targets specific CUDA versions.
- `dotnet build dotnet/CudaRS.sln -c Release` builds the .NET solution.
- `dotnet run --project dotnet/CudaRS.Examples/CudaRS.Examples.csproj` runs the sample app.
- `cargo test` runs Rust unit tests (see testing notes below).

## Coding Style & Naming Conventions
- Rust edition is 2021; follow `rustfmt` defaults and standard Rust naming.
- Raw bindings use `*-sys` crate names; safe wrappers mirror CUDA library names.
- Rust modules and functions use `snake_case`; public types use `UpperCamelCase`.
- C# namespaces are `CudaRS.*` and classes use `PascalCase`.
- Rust and C# must compile with zero warnings and zero errors; remove unused variables.
- C# must use strong, explicit types; avoid anonymous types and designs that rely on boxing/unboxing.

## Testing Guidelines
- Unit tests live beside code (for example `cuda-rs/*/src/lib.rs` and `build-support/cuda-build/src/lib.rs`).
- Run targeted tests with `cargo test -p cuda-runtime` (replace with any crate name).
- Tests rely on CUDA libraries/drivers being available on the machine.
- There are no dedicated `dotnet` test projects yet.

## FFI Contract Standard (Multi-language)
- Maintain a single, stable C ABI as the external contract; all languages use thin wrappers only.
- Export with `extern "C"` plus stable symbols (use `#[no_mangle]` or `#[unsafe(no_mangle)]` per toolchain rules).
- FFI types must be `#[repr(C)]` (or explicit integer repr). Expose opaque handles (`SdkHandle*`) with `create/destroy`.
- Memory/strings: prefer caller-allocated `*_len` + `*_write(buf, cap, out_written)`; if SDK allocates, provide `sdk_free(ptr, len)`.
- Errors: return `SdkErr` codes; expose `sdk_last_error_message_utf8()` for details. No panic/exception across the boundary (wrap with `catch_unwind`).
- Versioning: provide `sdk_abi_version()` and `sdk_version_string()`; document thread-safety and callback thread model.
- Tooling: generate `sdk.h` via `cbindgen`, package with `cargo-c`, and generate C# P/Invoke via `csbindgen`.
- Runtime model: keep `sdk-core` runtime-agnostic; FFI layer selects or embeds a runtime via features.

## Commit & Pull Request Guidelines
- Commit messages are imperative and sentence case (e.g., "Add TensorRT support").
- Keep commits focused; prefer one change theme per commit.
- PRs should include: summary, commands run, CUDA version/OS, and screenshots or sample output when changing examples or APIs.
- Link relevant issues or discussions when applicable.

## Configuration Tips
- Requires CUDA Toolkit 11.x or 12.x, Rust 1.70+, and .NET 8.0+.
- Use feature flags like `cuda-11`, `cuda-12`, `cuda-12-3`, and `runtime-linking` to match target toolkits.

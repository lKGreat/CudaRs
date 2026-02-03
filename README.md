# CudaRS

Rust CUDA bindings with a stable C ABI (sdk_*) and .NET wrappers. The stack supports multi-model loading and multiple pipelines per model, with high-level YOLO integrations.

## Project Structure

```
cudars/
  build-support/cuda-build/    # Shared CUDA detection/linking
  cuda-sys/                    # Raw FFI crates (-sys)
  cuda-rs/                     # Safe Rust wrappers
  cudars-core/                 # Runtime-agnostic core contracts
  cudars-ffi/                  # C ABI exports (cdylib)
  dotnet/                      # C# solutions
    CudaRS.Native/             # P/Invoke bindings (sdk_*)
    CudaRS.Core/               # SafeHandle + interop utilities
    CudaRS/                    # High-level API (ModelHub, PipelineBuilder)
    CudaRS.Yolo/               # YOLO models and pipelines
    CudaRS.Examples/           # Example app
```

## Build

```
# Rust (default CUDA 12 feature set)
cargo build --release

# Rust for CUDA 11 or CUDA 12.3
cargo build --release --features cuda-11
cargo build --release --features cuda-12-3

# .NET solution
dotnet build dotnet/CudaRS.sln -c Release
```

## Example (C# YOLO)

```csharp
using CudaRS.Yolo;

var enginePath = @"D:\\models\\yolo.engine";
var config = new YoloConfig
{
    Version = YoloVersion.V8,
    Task = YoloTask.Detect,
    InputWidth = 640,
    InputHeight = 640,
};

YoloVersionAdapter.ApplyVersionDefaults(config);

using var model = new YoloV8Model("yolo-v8", enginePath, config, deviceId: 0);
using var pipeline = new YoloGpuThroughputPipeline(model);

var imageBytes = File.ReadAllBytes("D:\\models\\test.jpg");
var result = await pipeline.EnqueueAsync(imageBytes, "demo", 0);
Console.WriteLine($"Detections: {result.Detections.Count}");
```

## FFI

The C ABI surface is exported from `cudars-ffi` with `sdk_*` symbols. The generated header is `cudars-ffi/include/sdk.h`.

## Requirements

- CUDA Toolkit 11.x or 12.x
- Rust 1.70+
- .NET 8.0+

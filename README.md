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

## PaddleOCR (Det/Rec/Cls + PP-Structure extension)

Enable the PaddleOCR integration with the `paddleocr` feature and provide the Paddle Inference + OpenCV locations:

```
set PADDLE_INFERENCE_ROOT=E:\codeding\AI\paddle_inference
set OPENCV_DIR=E:\codeding\AI\opencv\build
set PADDLE_OCR_ROOT=E:\codeding\AI\PaddleOCR-3.3.2
set ABSL_ROOT=E:\codeding\AI\abseil-cpp
```

Then build:

```
cargo build -p cudars-ffi --features paddleocr
```

OCR model configuration is passed as JSON via `SdkModelSpec.config_json`, with `det_model_dir` and `rec_model_dir` required. Pipeline config supports `enable_struct_json` to return structured JSON output.

## Requirements

- CUDA Toolkit 11.x or 12.x
- Rust 1.70+
- .NET 8.0+

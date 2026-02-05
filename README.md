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

# Rust with OpenVINO backend
cargo build --release --features openvino

# .NET solution
dotnet build dotnet/CudaRS.sln -c Release
```

## Fluent 单提供者链式 API（统一 Run/RunAsync）

示例：TensorRT YOLO，显式开启高吞吐与队列背压

```csharp
using CudaRS.Fluent;
using CudaRS.Yolo;

var pipeline = CudaRsFluent.Create()
    .Pipeline()
    .ForYolo(@"D:\models\yolo11.onnx", cfg =>
    {
        cfg.InputWidth = 640;
        cfg.InputHeight = 640;
    })
    .AsTensorRt() // 只能选一个提供者：TensorRt / OpenVino / Cpu / Paddle / Onnx
    .WithThroughput(t => { t.Enable = true; t.BatchSize = 8; t.NumStreams = 2; })
    .WithQueue(q => { q.Capacity = 64; q.TimeoutMs = -1; q.Backpressure = true; })
    .BuildYolo(); // 返回 IFluentImagePipeline<ModelInferenceResult>

var imageBytes = File.ReadAllBytes(@"D:\images\cat.jpg");
var result = await pipeline.RunAsync(imageBytes);
Console.WriteLine($"Detections: {result.Detections.Count}");
```

要点：
- 所有后端/设备共享同一调用签名，`As*` 仅用于选择唯一提供者，不做自动回退。
- `WithThroughput`/`WithQueue` 在构建阶段配置高吞吐和背压策略；若后端不支持将抛异常。

示例：PaddleOCR（需 `AsPaddle()`）

```csharp
using CudaRS.Fluent;
using CudaRS.Ocr;

var ocr = CudaRsFluent.Create()
    .Pipeline()
    .ForOcr(cfg =>
    {
        cfg.DetModelDir = @"D:\ocr\det";
        cfg.RecModelDir = @"D:\ocr\rec";
        cfg.Lang = "ch";
    })
    .AsPaddle()
    .WithQueue(q => { q.Capacity = 32; q.TimeoutMs = -1; })
    .BuildOcr(); // IFluentImagePipeline<OcrResult>

var bytes = File.ReadAllBytes(@"D:\images\doc.jpg");
var res = await ocr.RunAsync(bytes);
Console.WriteLine($"Lines: {res.Lines.Count}");
```

示例：OpenVINO Tensor（CPU/GPU 二选一，通用张量输入）

```csharp
using CudaRS.Fluent;
using CudaRS.OpenVino;

var tensor = CudaRsFluent.Create()
    .Pipeline()
    .ForTensor(@"D:\models\ov.xml",
        model => { model.ModelPath = @"D:\models\ov.xml"; },
        pipe => { pipe.OpenVinoDevice = "gpu"; })
    .AsOpenVino()            // 或 AsCpu()
    .WithThroughput(t => { t.Enable = true; t.NumStreams = 4; })
    .BuildTensor();          // IFluentTensorPipeline<OpenVinoTensorOutput[]>

float[] input = LoadInput();  // shape = [1,3,640,640]
long[] shape = { 1, 3, 640, 640 };
var outputs = await tensor.RunAsync(input, shape);
Console.WriteLine($"Outputs: {outputs.Length}");
```

提供者矩阵（单选）
- `AsTensorRt`: YOLO GPU (TensorRT)
- `AsOnnx`: YOLO GPU/CPU 取决于 ONNX Runtime 构建
- `AsOpenVino`: YOLO OpenVINO GPU / 通用 Tensor OpenVINO GPU
- `AsCpu`: YOLO OpenVINO CPU / 通用 Tensor OpenVINO CPU
- `AsPaddle`: OCR (PaddleOCR)

## OpenVINO

OpenVINO can be selected as a runtime device for YOLO and via a generic tensor pipeline. OpenVINO options are passed as a JSON object string. You can also set higher-level fields for performance mode, request pool size, cache dir, and device overrides (e.g., `NVIDIA` via the plugin). Note: NVIDIA requires the OpenVINO NVIDIA plugin; it is not enabled by default.

```csharp
using CudaRS.Yolo;

var options = new YoloPipelineOptions
{
    Device = InferenceDevice.OpenVino,
    OpenVinoDevice = "gpu",
    OpenVinoPerformanceMode = "throughput",
    OpenVinoNumRequests = 8,
    OpenVinoCacheDir = ".\\ov_cache"
};
```

Generic OpenVINO usage:

```csharp
using CudaRS.OpenVino;

var model = new OpenVinoModel("generic", new OpenVinoModelConfig
{
    ModelPath = @"D:\\models\\model.xml"
});
using var pipeline = model.CreatePipeline("default", new OpenVinoPipelineConfig
{
    OpenVinoDevice = "cpu",
    OpenVinoConfigJson = "{\"NUM_STREAMS\":\"2\"}"
});

// Direct OpenVINO native usage (device overrides).
using var native = new OpenVinoNativeModel(
    @"D:\\models\\model.xml",
    new OpenVinoNativeConfig
    {
        Device = "auto",
        DeviceName = "AUTO:GPU,CPU",
        PerformanceMode = "throughput",
        NumRequests = 8
    });
```

## PaddlePaddle Models with OpenVINO

CudaRS supports loading PaddlePaddle models (including the newer `.json` format used by PP-OCRv5) through conversion to ONNX and inference with OpenVINO.

### Quick Start

```csharp
using CudaRS.OpenVino;
using CudaRS.Paddle;

// 1. Convert PaddlePaddle model to ONNX (uses caching)
var converter = new Paddle2OnnxConverter();
var onnxPath = converter.ConvertOrUseCache(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.json",
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.pdiparams"
);

// 2. Load with OpenVINO
var config = new OpenVinoModelConfig { ModelPath = onnxPath };
using var model = new OpenVinoModel("paddle_det", config);
using var pipeline = model.CreatePipeline("CPU");

// 3. Run inference
var outputs = pipeline.Run(inputData, inputShape);
```

### Prerequisites

Install Python and paddle2onnx:

```bash
pip install paddle2onnx onnx onnxruntime
```

### Conversion Scripts

Convert models using provided scripts:

**Python:**
```bash
python scripts/paddle2onnx_converter.py \
  --model_dir E:\models\PP-OCRv5_mobile_det_infer \
  --output model.onnx
```

**PowerShell:**
```powershell
.\scripts\convert_paddle_models.ps1 `
  -ModelDir "E:\models\PP-OCRv5_mobile_det_infer" `
  -OutputPath "model.onnx"
```

### Preprocessing Configuration

Load preprocessing settings from `inference.yml`:

```csharp
var preprocessConfig = PaddlePreprocessConfig.FromYaml(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.yml"
);
var preprocessed = preprocessConfig.Preprocess(imageData, 3, 640, 640);
```

### Complete Guide

See [docs/PADDLE_OPENVINO_GUIDE.md](docs/PADDLE_OPENVINO_GUIDE.md) for detailed documentation, including:
- Model conversion workflows
- Preprocessing configuration
- Performance optimization
- Troubleshooting
- Complete examples

### Example

Run the complete example:

```csharp
CasePaddleOpenVinoTest.Run();  // See dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs
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
If you want PP-Structure style outputs, provide a PaddleOCR pipeline config YAML via `paddlex_config_yaml` and enable `enable_struct_json` in the pipeline config.
For PaddleOCR + OpenVINO, set `device` to `openvino` in the model config. The optional `openvino_config_json` field is passed through for future OpenVINO property support.

## Requirements

- CUDA Toolkit 11.x or 12.x
- Rust 1.70+
- .NET 8.0+

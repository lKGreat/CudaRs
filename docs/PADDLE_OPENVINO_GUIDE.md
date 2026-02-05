# PaddlePaddle Models with OpenVINO - Complete Guide

This guide explains how to use PaddlePaddle models with OpenVINO in the CudaRS project.

## Table of Contents

1. [Overview](#overview)
2. [PaddlePaddle Model Formats](#paddlepaddle-model-formats)
3. [Installation and Setup](#installation-and-setup)
4. [Model Conversion](#model-conversion)
5. [C# Usage](#csharp-usage)
6. [Rust Usage](#rust-usage)
7. [Preprocessing Configuration](#preprocessing-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Overview

### Why Convert PaddlePaddle to ONNX?

OpenVINO natively supports ONNX format but has limited support for newer PaddlePaddle model formats (`.json` based). By converting PaddlePaddle models to ONNX, you can:

- **Use OpenVINO's optimizations**: Leverage OpenVINO's hardware acceleration and model optimization capabilities
- **Broader hardware support**: Deploy on Intel CPUs, GPUs, VPUs, and other OpenVINO-supported devices
- **Simplified deployment**: Use a unified inference framework across different model sources

### Conversion Workflow

```
PaddlePaddle Model → ONNX → OpenVINO → Inference
(inference.json)     (.onnx)  (compiled)  (results)
```

## PaddlePaddle Model Formats

### New Format (PaddlePaddle 3.0+)

Newer PaddlePaddle models (like PP-OCRv5) use a JSON-based format:

- **`inference.json`**: Model structure in JSON format
- **`inference.pdiparams`**: Model weights/parameters
- **`inference.yml`**: Preprocessing configuration (optional)

### Legacy Format

Older PaddlePaddle models use:

- **`.pdmodel`**: Model structure in protobuf format
- **`.pdiparams`**: Model weights/parameters

### Conversion Requirements

Both formats can be converted to ONNX, but the conversion commands differ slightly:

- **JSON format**: Requires specifying `--model_filename inference.json`
- **PDModel format**: Can often auto-detect the model structure file

## Installation and Setup

### Prerequisites

1. **Python 3.7 or higher**
   ```bash
   python --version
   ```

2. **Install conversion tools**
   ```bash
   pip install paddle2onnx onnx onnxruntime
   ```

3. **Verify installation**
   ```bash
   python -c "import paddle2onnx; print('paddle2onnx installed')"
   python -c "import onnx; print('onnx installed')"
   ```

### Project Setup

Ensure you have the conversion scripts in your project:

```
cudars/
├── scripts/
│   ├── paddle2onnx_converter.py   # Python conversion script
│   └── convert_paddle_models.ps1  # PowerShell batch converter
└── docs/
    └── PADDLE_OPENVINO_GUIDE.md   # This guide
```

## Model Conversion

### Method 1: Using Python Script (Recommended)

#### Single Model Conversion

```bash
python scripts/paddle2onnx_converter.py \
  --model_dir E:\models\PP-OCRv5_mobile_det_infer \
  --output model_det.onnx \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --opset_version 11
```

#### Batch Conversion

Convert multiple models at once:

```bash
python scripts/paddle2onnx_converter.py \
  --batch \
  --input_dir E:\models\paddle_models \
  --output_dir E:\models\onnx_models \
  --opset_version 11
```

#### Options

- `--model_dir`: Directory containing the PaddlePaddle model
- `--output`: Path to save the ONNX model
- `--model_filename`: Model structure filename (default: `inference.json`)
- `--params_filename`: Parameters filename (default: `inference.pdiparams`)
- `--opset_version`: ONNX opset version (default: 11, recommended: 11-13)
- `--no-validation`: Skip ONNX model validation (faster but less safe)

### Method 2: Using PowerShell Script

#### Single Model Conversion

```powershell
.\scripts\convert_paddle_models.ps1 `
  -ModelDir "E:\models\PP-OCRv5_mobile_det_infer" `
  -OutputPath "model_det.onnx" `
  -OpsetVersion 11
```

#### Batch Conversion

```powershell
.\scripts\convert_paddle_models.ps1 `
  -InputDir "E:\models\paddle_models" `
  -OutputDir "E:\models\onnx_models" `
  -OpsetVersion 11
```

#### Help

```powershell
.\scripts\convert_paddle_models.ps1 -Help
```

### Method 3: Direct Command Line

Using `paddle2onnx` CLI:

```bash
paddle2onnx \
  --model_dir ./PP-OCRv5_mobile_det_infer \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file model.onnx \
  --opset_version 11
```

## C# Usage

### Basic Example

```csharp
using CudaRS.OpenVino;
using CudaRS.Paddle;

// 1. Convert PaddlePaddle model to ONNX (or use cached)
var converter = new Paddle2OnnxConverter();
var onnxPath = converter.ConvertOrUseCache(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.json",
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.pdiparams",
    opsetVersion: 11
);

// 2. Load with OpenVINO
var modelConfig = new OpenVinoModelConfig { ModelPath = onnxPath };
using var model = new OpenVinoModel("paddle_det", modelConfig);
using var pipeline = model.CreatePipeline("CPU");

// 3. Prepare input (replace with real image data)
var inputData = new float[1 * 3 * 640 * 640];
var inputShape = new long[] { 1, 3, 640, 640 };

// 4. Run inference
var outputs = pipeline.Run(inputData, inputShape);

// 5. Process outputs
foreach (var output in outputs)
{
    Console.WriteLine($"Output shape: [{string.Join(", ", output.Shape)}]");
    Console.WriteLine($"Output size: {output.Data.Length}");
}
```

### Native SDK Conversion (Bundled Python)

Convert PaddleOCR det/rec models to OpenVINO IR via the stable C ABI and bundled Python runtime:

```csharp
using CudaRS.Paddle;

var converter = new PaddleToIrConverter();
var (detXml, recXml) = converter.Convert(
    detModelDir: @"E:\models\PP-OCRv5_mobile_det_infer",
    recModelDir: @"E:\models\PP-OCRv5_mobile_rec_infer",
    outputDir: @"E:\models\ov_ir",
    options: new PaddleOcrIrOptions
    {
        OpsetVersion = 11,
        CompressToFp16 = true,
        TimeoutSeconds = 300
    });
```

### With Preprocessing Configuration

```csharp
using CudaRS.Paddle;

// Load preprocessing config from inference.yml
var preprocessConfig = PaddlePreprocessConfig.FromYaml(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.yml"
);

Console.WriteLine($"Preprocessing: {preprocessConfig}");

// Apply preprocessing to image data
var preprocessed = preprocessConfig.Preprocess(
    imageData, 
    channels: 3, 
    height: 640, 
    width: 640
);
```

### Advanced: Batch Conversion

```csharp
var converter = new Paddle2OnnxConverter();

// Convert detection model
var detOnnx = converter.ConvertDirectory(
    @"E:\models\PP-OCRv5_mobile_det_infer",
    opsetVersion: 11
);

// Convert recognition model
var recOnnx = converter.ConvertDirectory(
    @"E:\models\PP-OCRv5_mobile_rec_infer",
    opsetVersion: 11
);

// Use both models
var detModel = new OpenVinoModel("det", new OpenVinoModelConfig { ModelPath = detOnnx });
var recModel = new OpenVinoModel("rec", new OpenVinoModelConfig { ModelPath = recOnnx });
```

### Cache Management

```csharp
var converter = new Paddle2OnnxConverter();

// Check cache size
var cacheSize = converter.GetCacheSize();
Console.WriteLine($"Cache size: {cacheSize / 1024.0:F2} KB");

// Clear cache
converter.ClearCache();

// Force reconversion (bypass cache)
var onnxPath = converter.ConvertOrUseCache(
    modelJsonPath, 
    modelParamsPath,
    forceReconvert: true
);
```

### Complete Example

See `dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs` for a complete working example.

## Rust Usage

### Basic Example

```rust
use std::process::Command;

// Convert PaddlePaddle to ONNX
let output = Command::new("python")
    .arg("scripts/paddle2onnx_converter.py")
    .arg("--model_dir").arg("E:/models/PP-OCRv5_mobile_det_infer")
    .arg("--output").arg("model.onnx")
    .arg("--opset_version").arg("11")
    .output()
    .expect("Failed to convert model");

// Load with OpenVINO (add OpenVINO code here)
// See cudars-ffi/src/openvino.rs for OpenVINO bindings
```

### Native SDK Conversion via FFI

```rust
use std::ffi::CString;

extern "C" {
    fn sdk_convert_paddle_ocr_to_ir(
        det_model_dir_ptr: *const i8,
        det_model_dir_len: usize,
        rec_model_dir_ptr: *const i8,
        rec_model_dir_len: usize,
        output_dir_ptr: *const i8,
        output_dir_len: usize,
        options_json_ptr: *const i8,
        options_json_len: usize,
        det_xml_buf: *mut i8,
        det_xml_cap: usize,
        det_xml_written: *mut usize,
        rec_xml_buf: *mut i8,
        rec_xml_cap: usize,
        rec_xml_written: *mut usize,
    ) -> i32;
}

let det = CString::new("E:/models/PP-OCRv5_mobile_det_infer").unwrap();
let rec = CString::new("E:/models/PP-OCRv5_mobile_rec_infer").unwrap();
let out = CString::new("E:/models/ov_ir").unwrap();
let opts = CString::new("{\"compress_to_fp16\":true}").unwrap();

let mut det_buf = vec![0i8; 1024];
let mut rec_buf = vec![0i8; 1024];
let mut det_written = 0usize;
let mut rec_written = 0usize;

let err = unsafe {
    sdk_convert_paddle_ocr_to_ir(
        det.as_ptr(),
        det.as_bytes().len(),
        rec.as_ptr(),
        rec.as_bytes().len(),
        out.as_ptr(),
        out.as_bytes().len(),
        opts.as_ptr(),
        opts.as_bytes().len(),
        det_buf.as_mut_ptr(),
        det_buf.len(),
        &mut det_written,
        rec_buf.as_mut_ptr(),
        rec_buf.len(),
        &mut rec_written,
    )
};

if err != 0 {
    panic!("conversion failed: {}", err);
}
```

### Complete Example

See `cudars-ffi/examples/paddle_to_openvino.rs` for a complete working example:

```bash
cargo run --example paddle_to_openvino --features openvino
```

## Preprocessing Configuration

### Understanding inference.yml

PaddlePaddle models often include an `inference.yml` file with preprocessing parameters:

```yaml
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
scale: 255.0
image_shape: [3, 640, 640]
is_scale: true
channel_order: CHW
color_space: RGB
```

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mean` | Mean values for normalization | `[0.485, 0.456, 0.406]` (ImageNet) |
| `std` | Standard deviation for normalization | `[0.229, 0.224, 0.225]` |
| `scale` | Scale factor before normalization | `255.0` (for 0-255 to 0-1) |
| `image_shape` | Expected input shape | `[3, 640, 640]` (C, H, W) |
| `is_scale` | Whether to apply scaling | `true` |
| `channel_order` | Channel ordering | `CHW` or `HWC` |
| `color_space` | Color space | `RGB` or `BGR` |

### Preprocessing Formula

```
normalized = (pixel_value * scale - mean) / std
```

For example with ImageNet normalization:
```
normalized_r = (r / 255.0 - 0.485) / 0.229
normalized_g = (g / 255.0 - 0.456) / 0.224
normalized_b = (b / 255.0 - 0.406) / 0.225
```

### Using Preprocessing in C#

```csharp
// Load from file
var config = PaddlePreprocessConfig.FromYaml("inference.yml");

// Or create manually
var config = new PaddlePreprocessConfig
{
    Mean = new[] { 0.485f, 0.456f, 0.406f },
    Std = new[] { 0.229f, 0.224f, 0.225f },
    Scale = 1.0f / 255.0f,
    ImageShape = new[] { 3, 640, 640 },
    IsCHW = true,
    ColorSpace = "RGB"
};

// Apply preprocessing
var preprocessed = config.Preprocess(imageData, 3, 640, 640);
```

## Performance Optimization

### OpenVINO Configuration

```csharp
var pipelineConfig = new OpenVinoPipelineConfig
{
    // Device selection
    OpenVinoDevice = "CPU",  // or "GPU", "AUTO"
    
    // Performance mode
    OpenVinoPerformanceMode = "throughput",  // or "latency"
    
    // Number of parallel streams
    OpenVinoNumStreams = 4,
    
    // Model caching
    OpenVinoCacheDir = "./ov_cache",
    
    // Memory-mapped loading
    OpenVinoEnableMmap = true,
    
    // Profiling
    OpenVinoEnableProfiling = false
};
```

### Performance Tips

1. **Use model caching**: Set `OpenVinoCacheDir` to cache compiled models
2. **Choose the right device**: 
   - `CPU`: Good for general purpose
   - `GPU`: Best for large models and batch inference
   - `AUTO`: Let OpenVINO choose automatically
3. **Adjust performance mode**:
   - `latency`: Minimize single-request latency
   - `throughput`: Maximize overall throughput
4. **Enable memory mapping**: Set `OpenVinoEnableMmap = true` for faster loading
5. **Optimize ONNX opset**: Try different opset versions (11, 13) for best compatibility

### Batch Inference

For better throughput, use batch inference:

```csharp
var inputs = new ReadOnlyMemory<float>[]
{
    imageData1,
    imageData2,
    imageData3,
    imageData4
};

var singleShape = new long[] { 3, 640, 640 };
var results = pipeline.RunBatch(inputs, singleShape);
```

## Troubleshooting

### Common Issues

#### 1. paddle2onnx not found

**Error:**
```
ModuleNotFoundError: No module named 'paddle2onnx'
```

**Solution:**
```bash
pip install paddle2onnx onnx onnxruntime
```

#### 2. Unsupported operators

**Error:**
```
NotImplementedError: Unsupported operator: XXX
```

**Solution:**
- Update paddle2onnx: `pip install --upgrade paddle2onnx`
- Try a different opset version: `--opset_version 13`
- Check PaddlePaddle model compatibility

#### 3. Model conversion fails

**Error:**
```
RuntimeError: Failed to convert model
```

**Solutions:**
- Verify model files exist and are not corrupted
- Check file permissions
- Ensure sufficient disk space
- Try with `--no-validation` flag (debugging only)

#### 4. Inference shape mismatch

**Error:**
```
Shape mismatch: expected [1, 3, 640, 640], got [1, 3, 480, 640]
```

**Solution:**
- Check input shape in model info: `model.GetInputInfo()`
- Resize image to match expected dimensions
- Use dynamic shape if model supports it

#### 5. Python not found

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solutions:**
- Install Python 3.7+ from python.org
- Add Python to PATH
- Or specify full path in converter: `new Paddle2OnnxConverter(pythonPath: @"C:\Python39\python.exe")`

### Debugging Tips

1. **Enable validation**: Remove `--no-validation` flag to check ONNX model integrity
2. **Check model info**: Print input/output shapes and data types
3. **Test with simple input**: Use random data first before real images
4. **Enable profiling**: Set `OpenVinoEnableProfiling = true` to identify bottlenecks
5. **Check logs**: Review console output for warnings and errors

## FAQ

### Q: Which ONNX opset version should I use?

**A:** Use opset 11 or 13 for best compatibility with OpenVINO. Newer opset versions may have limited support.

### Q: Can I convert all PaddlePaddle models?

**A:** Most models can be converted, but some custom operators may not be supported. Check paddle2onnx compatibility list.

### Q: Is there any accuracy loss after conversion?

**A:** Generally minimal. Differences are usually due to:
- Floating-point precision differences
- Different optimization strategies
- Always validate with test data

### Q: How do I handle dynamic input shapes?

**A:** 
1. Check if model supports dynamic shapes with `model.GetInputInfo()`
2. For OpenVINO, use reshape before inference
3. Or convert with specific input shapes

### Q: Should I cache converted models?

**A:** Yes! The `Paddle2OnnxConverter` automatically caches conversions. This saves time on subsequent runs.

### Q: Can I deploy without Python?

**A:** Yes! After conversion:
1. Convert once during development/CI
2. Include ONNX model in deployment
3. Only OpenVINO runtime needed in production (no Python required)

### Q: How to optimize for production?

**A:**
1. Convert models during build, not runtime
2. Use OpenVINO model caching (`OpenVinoCacheDir`)
3. Enable memory mapping (`OpenVinoEnableMmap`)
4. Choose appropriate performance mode
5. Consider INT8 quantization for better performance

### Q: What about PP-OCR end-to-end inference?

**A:** Convert both detection and recognition models separately, then chain them:

```csharp
var detModel = converter.ConvertDirectory("det_model_dir");
var recModel = converter.ConvertDirectory("rec_model_dir");

// Run detection
var detOutputs = detPipeline.Run(image, ...);

// Extract text regions from detection
// Run recognition on each region
foreach (var region in detRegions)
{
    var recOutputs = recPipeline.Run(region, ...);
    // Decode text
}
```

## Additional Resources

- [PaddlePaddle Official Documentation](https://www.paddlepaddle.org.cn/)
- [Paddle2ONNX GitHub](https://github.com/PaddlePaddle/Paddle2ONNX)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

## Support

For issues and questions:
- Check this guide and troubleshooting section
- Review example code in `dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs`
- Check project issues on GitHub
- Consult PaddlePaddle and OpenVINO communities

---

**Last Updated:** 2026-02-05

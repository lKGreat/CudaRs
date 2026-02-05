# PaddlePaddle Models Testing Guide

This guide provides instructions for testing PaddlePaddle model conversion and inference with OpenVINO.

## Test Checklist

### 1. Environment Setup Tests

- [ ] Python 3.7+ installed
- [ ] paddle2onnx package installed
- [ ] onnx package installed
- [ ] onnxruntime package installed
- [ ] Conversion scripts accessible

**Verification Commands:**
```bash
python --version
python -c "import paddle2onnx; print(paddle2onnx.__version__)"
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

### 2. Conversion Script Tests

#### Test 1: Single Model Conversion (Python)

```bash
python scripts/paddle2onnx_converter.py \
  --model_dir <path-to-paddle-model> \
  --output test_output.onnx \
  --opset_version 11
```

**Expected Result:**
- ✓ Conversion completes without errors
- ✓ ONNX file created at specified output path
- ✓ ONNX validation passes
- ✓ Input/output shapes displayed

#### Test 2: Single Model Conversion (PowerShell)

```powershell
.\scripts\convert_paddle_models.ps1 `
  -ModelDir "<path-to-paddle-model>" `
  -OutputPath "test_output.onnx"
```

**Expected Result:**
- ✓ Dependencies checked successfully
- ✓ Conversion completes
- ✓ Output file exists

#### Test 3: Batch Conversion

```bash
python scripts/paddle2onnx_converter.py \
  --batch \
  --input_dir <dir-with-multiple-models> \
  --output_dir <output-dir>
```

**Expected Result:**
- ✓ All models discovered
- ✓ All models converted
- ✓ Success count matches total count

### 3. C# Converter Tests

#### Test 1: Basic Conversion

```csharp
var converter = new Paddle2OnnxConverter();
var onnxPath = converter.ConvertOrUseCache(
    "path/to/inference.json",
    "path/to/inference.pdiparams"
);
Assert.IsTrue(File.Exists(onnxPath));
```

#### Test 2: Cache Functionality

```csharp
var converter = new Paddle2OnnxConverter();

// First conversion
var sw = Stopwatch.StartNew();
var path1 = converter.ConvertOrUseCache(jsonPath, paramsPath);
var firstTime = sw.ElapsedMilliseconds;

// Second conversion (should use cache)
sw.Restart();
var path2 = converter.ConvertOrUseCache(jsonPath, paramsPath);
var secondTime = sw.ElapsedMilliseconds;

// Cache should be much faster
Assert.IsTrue(secondTime < firstTime / 10);
Assert.AreEqual(path1, path2);
```

#### Test 3: Force Reconversion

```csharp
var converter = new Paddle2OnnxConverter();
var path1 = converter.ConvertOrUseCache(jsonPath, paramsPath);
var path2 = converter.ConvertOrUseCache(jsonPath, paramsPath, forceReconvert: true);
Assert.AreEqual(path1, path2);
// Verify file timestamp updated
```

### 4. Preprocessing Configuration Tests

#### Test 1: YAML Parsing

```csharp
var config = PaddlePreprocessConfig.FromYaml("inference.yml");
Assert.IsNotNull(config.Mean);
Assert.IsNotNull(config.Std);
Assert.IsTrue(config.ImageShape?.Length > 0);
```

#### Test 2: Preprocessing Application

```csharp
var config = new PaddlePreprocessConfig
{
    Mean = new[] { 0.5f, 0.5f, 0.5f },
    Std = new[] { 0.5f, 0.5f, 0.5f },
    Scale = 1.0f / 255.0f
};

var input = new float[3 * 224 * 224];
Array.Fill(input, 128.0f);  // Fill with 128

var output = config.Preprocess(input, 3, 224, 224);

// Check normalization applied correctly
// (128 / 255 - 0.5) / 0.5 = 0.003921...
Assert.IsTrue(Math.Abs(output[0] - 0.00392) < 0.01);
```

### 5. Model-Specific Tests

#### PP-OCRv5 Detection Model

**Model Files:**
- `inference.json`
- `inference.pdiparams`
- `inference.yml`

**Test Steps:**
1. Convert model
2. Load with OpenVINO
3. Verify input shape: `[1, 3, 640, 640]` or dynamic
4. Run inference with random input
5. Check output shape matches detection format

**Expected Outputs:**
- Detection boxes
- Confidence scores

#### PP-OCRv5 Recognition Model

**Model Files:**
- `inference.json`
- `inference.pdiparams`
- `inference.yml`

**Test Steps:**
1. Convert model
2. Load with OpenVINO
3. Verify input shape: `[1, 3, 48, 320]` or similar
4. Run inference
5. Validate output format

**Expected Outputs:**
- Character probabilities
- Sequence predictions

#### PaddleClas Classification Model

**Test Steps:**
1. Convert ImageNet classification model
2. Load with OpenVINO
3. Input shape: `[1, 3, 224, 224]`
4. Run inference
5. Check top-5 predictions

**Expected Outputs:**
- Class probabilities (1000 classes for ImageNet)

#### PaddleDetection Model (YOLO, etc.)

**Test Steps:**
1. Convert detection model
2. Load with OpenVINO
3. Typical input: `[1, 3, 640, 640]`
4. Run inference
5. Parse detection outputs

**Expected Outputs:**
- Bounding boxes
- Class IDs
- Confidence scores

### 6. Integration Tests

#### Test 1: End-to-End Pipeline

```csharp
// Convert
var converter = new Paddle2OnnxConverter();
var onnxPath = converter.ConvertDirectory(paddleModelDir);

// Load config
var preprocessConfig = PaddlePreprocessConfig.FromYaml(
    Path.Combine(paddleModelDir, "inference.yml")
);

// Create model
var modelConfig = new OpenVinoModelConfig { ModelPath = onnxPath };
using var model = new OpenVinoModel("test", modelConfig);
using var pipeline = model.CreatePipeline("CPU");

// Prepare input
var inputData = GenerateRandomInput(preprocessConfig);
var preprocessed = preprocessConfig.Preprocess(inputData, 3, 640, 640);

// Inference
var outputs = pipeline.Run(preprocessed, new long[] { 1, 3, 640, 640 });

// Validate
Assert.IsTrue(outputs.Length > 0);
Assert.IsTrue(outputs[0].Data.Length > 0);
```

#### Test 2: Batch Inference

```csharp
var inputs = new ReadOnlyMemory<float>[]
{
    GenerateRandomInput(1, 3, 640, 640),
    GenerateRandomInput(1, 3, 640, 640),
    GenerateRandomInput(1, 3, 640, 640),
    GenerateRandomInput(1, 3, 640, 640)
};

var results = pipeline.RunBatch(inputs, new long[] { 3, 640, 640 });
Assert.AreEqual(4, results.Length);
```

#### Test 3: Multiple Models

```csharp
// Convert detection and recognition models
var detOnnx = converter.ConvertDirectory(detModelDir);
var recOnnx = converter.ConvertDirectory(recModelDir);

// Load both models
var detModel = new OpenVinoModel("det", new OpenVinoModelConfig { ModelPath = detOnnx });
var recModel = new OpenVinoModel("rec", new OpenVinoModelConfig { ModelPath = recOnnx });

// Use both in pipeline
using var detPipeline = detModel.CreatePipeline("CPU");
using var recPipeline = recModel.CreatePipeline("CPU");

// Run detection, then recognition
var detOutputs = detPipeline.Run(...);
// Extract regions, then:
var recOutputs = recPipeline.Run(...);
```

### 7. Performance Tests

#### Test 1: Conversion Time

Measure time to convert various model sizes:
- Small model (~5MB): Expected < 5s
- Medium model (~50MB): Expected < 30s
- Large model (~100MB+): Expected < 60s

#### Test 2: Inference Latency

Measure single inference time:
- CPU (latency mode): Record baseline
- CPU (throughput mode): Compare
- GPU (if available): Compare

#### Test 3: Throughput

Measure inferences per second:
- Single stream
- Multiple streams
- Batch inference

### 8. Error Handling Tests

#### Test 1: Missing Files

```csharp
Assert.ThrowsException<FileNotFoundException>(() => 
    converter.ConvertOrUseCache("nonexistent.json", "nonexistent.pdiparams")
);
```

#### Test 2: Invalid Model Files

Test with corrupted or invalid model files.

**Expected:** Conversion should fail with clear error message.

#### Test 3: Unsupported Operators

Test with models containing operators not supported by paddle2onnx.

**Expected:** Conversion should report which operators are unsupported.

### 9. Compatibility Tests

Test with different model formats:

- [ ] PaddlePaddle 3.0+ (inference.json format)
- [ ] PaddlePaddle 2.x (inference.pdmodel format)
- [ ] Different opset versions (9, 11, 13)
- [ ] Dynamic vs static shapes
- [ ] Different data types (FP32, FP16, INT8)

### 10. Cleanup Tests

#### Test 1: Cache Management

```csharp
var converter = new Paddle2OnnxConverter();

// Convert some models
converter.ConvertOrUseCache(json1, params1);
converter.ConvertOrUseCache(json2, params2);

// Check cache size
var sizeBeforeClear = converter.GetCacheSize();
Assert.IsTrue(sizeBeforeClear > 0);

// Clear cache
converter.ClearCache();

// Verify cleared
var sizeAfterClear = converter.GetCacheSize();
Assert.AreEqual(0, sizeAfterClear);
```

## Test Data

### Required Test Models

1. **PP-OCRv5 Detection Model**
   - Download: [PaddleOCR Models](https://github.com/PaddlePaddle/PaddleOCR)
   - Size: ~5MB
   - Format: inference.json + inference.pdiparams

2. **PP-OCRv5 Recognition Model**
   - Download: Same as above
   - Size: ~10MB
   - Format: inference.json + inference.pdiparams

3. **PaddleClas ImageNet Model**
   - Download: [PaddleClas Models](https://github.com/PaddlePaddle/PaddleClas)
   - Size: ~25MB
   - Format: May be .pdmodel or .json

4. **PaddleDetection YOLO Model**
   - Download: [PaddleDetection Models](https://github.com/PaddlePaddle/PaddleDetection)
   - Size: Varies
   - Format: Varies

### Test Images

Prepare test images for inference validation:
- Document images (for OCR)
- Natural images (for classification)
- Object detection images (for detection)

## Running Tests

### Run All C# Tests

```bash
cd dotnet/CudaRS.Examples
dotnet run
```

Then select the PaddlePaddle-OpenVINO test case.

### Run Specific Test

```csharp
CasePaddleOpenVinoTest.Run();
```

### Run Rust Example

```bash
cargo run --example paddle_to_openvino --features openvino
```

## Expected Results Summary

| Test Category | Expected Success Rate |
|---------------|----------------------|
| Environment Setup | 100% |
| Conversion Scripts | 100% |
| C# Converter | 100% |
| Preprocessing | 100% |
| Model-Specific | 90%+ (some models may have unsupported ops) |
| Integration | 100% |
| Performance | Baseline established |
| Error Handling | 100% |
| Compatibility | 80%+ (newer models may have issues) |
| Cleanup | 100% |

## Troubleshooting Test Failures

### Conversion Failures
- Verify paddle2onnx version
- Check model file integrity
- Try different opset versions

### Inference Failures
- Verify input shapes
- Check preprocessing configuration
- Enable OpenVINO profiling for details

### Performance Issues
- Enable caching
- Use appropriate device (CPU/GPU)
- Optimize OpenVINO configuration

## Continuous Testing

For CI/CD, consider:
1. Automated conversion tests with sample models
2. Performance regression tests
3. Compatibility tests with new paddle2onnx versions
4. Memory leak tests for long-running scenarios

## Reporting Issues

When reporting test failures, include:
- Model source and version
- paddle2onnx version
- ONNX opset version used
- Error messages and stack traces
- System information (OS, Python version, etc.)

---

**Last Updated:** 2026-02-05

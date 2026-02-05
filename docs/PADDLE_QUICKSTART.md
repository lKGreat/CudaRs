# PaddlePaddle Models with OpenVINO - Quick Start

快速开始使用 PaddlePaddle 模型和 OpenVINO 推理。

## 5 分钟快速开始

### 步骤 1: 安装依赖

```bash
# 安装 Python 转换工具
pip install paddle2onnx onnx onnxruntime
```

### 步骤 2: 准备 PaddlePaddle 模型

确保你有 PaddlePaddle 模型文件：
- `inference.json` - 模型结构
- `inference.pdiparams` - 模型参数
- `inference.yml` - 预处理配置（可选）

### 步骤 3: 转换模型

**使用 PowerShell:**
```powershell
.\scripts\convert_paddle_models.ps1 `
  -ModelDir "E:\models\PP-OCRv5_mobile_det_infer" `
  -OutputPath "model.onnx"
```

**使用 Python:**
```bash
python scripts/paddle2onnx_converter.py \
  --model_dir E:\models\PP-OCRv5_mobile_det_infer \
  --output model.onnx
```

### 步骤 4: C# 推理

```csharp
using CudaRS.OpenVino;
using CudaRS.Paddle;

// 自动转换并加载
var converter = new Paddle2OnnxConverter();
var onnxPath = converter.ConvertOrUseCache(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.json",
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.pdiparams"
);

// 创建 OpenVINO 模型
var config = new OpenVinoModelConfig { ModelPath = onnxPath };
using var model = new OpenVinoModel("det", config);
using var pipeline = model.CreatePipeline("CPU");

// 推理
var inputData = new float[1 * 3 * 640 * 640]; // 你的输入数据
var outputs = pipeline.Run(inputData, new long[] { 1, 3, 640, 640 });

Console.WriteLine($"检测到 {outputs.Length} 个输出");
```

### 步骤 5: 运行示例

```bash
cd dotnet/CudaRS.Examples
dotnet run
# 选择 PaddlePaddle-OpenVINO 测试用例
```

## 常见模型示例

### PP-OCRv5 文字检测

```csharp
var converter = new Paddle2OnnxConverter();
var detOnnx = converter.ConvertDirectory(
    @"E:\models\PP-OCRv5_mobile_det_infer"
);

var model = new OpenVinoModel("det", new OpenVinoModelConfig { ModelPath = detOnnx });
var pipeline = model.CreatePipeline("CPU");
```

### PP-OCRv5 文字识别

```csharp
var recOnnx = converter.ConvertDirectory(
    @"E:\models\PP-OCRv5_mobile_rec_infer"
);

var model = new OpenVinoModel("rec", new OpenVinoModelConfig { ModelPath = recOnnx });
var pipeline = model.CreatePipeline("CPU");
```

### PaddleClas 图像分类

```csharp
var clsOnnx = converter.ConvertDirectory(
    @"E:\models\ResNet50_infer"
);

var model = new OpenVinoModel("cls", new OpenVinoModelConfig { ModelPath = clsOnnx });
var pipeline = model.CreatePipeline("CPU");
```

## 性能优化

### 使用 GPU 加速

```csharp
var pipelineConfig = new OpenVinoPipelineConfig
{
    OpenVinoDevice = "GPU",
    OpenVinoPerformanceMode = "throughput"
};
var pipeline = model.CreatePipeline(pipelineConfig);
```

### 批量推理

```csharp
var inputs = new ReadOnlyMemory<float>[]
{
    imageData1,
    imageData2,
    imageData3,
    imageData4
};
var results = pipeline.RunBatch(inputs, new long[] { 3, 640, 640 });
```

### 启用模型缓存

```csharp
var pipelineConfig = new OpenVinoPipelineConfig
{
    OpenVinoDevice = "CPU",
    OpenVinoCacheDir = "./ov_cache",  // 模型编译缓存
    OpenVinoEnableMmap = true  // 内存映射加载
};
```

## 预处理配置

### 从 YAML 加载

```csharp
var preprocessConfig = PaddlePreprocessConfig.FromYaml(
    @"E:\models\PP-OCRv5_mobile_det_infer\inference.yml"
);

// 应用预处理
var preprocessed = preprocessConfig.Preprocess(imageData, 3, 640, 640);
```

### 手动配置

```csharp
var preprocessConfig = new PaddlePreprocessConfig
{
    Mean = new[] { 0.485f, 0.456f, 0.406f },
    Std = new[] { 0.229f, 0.224f, 0.225f },
    Scale = 1.0f / 255.0f,
    ImageShape = new[] { 3, 640, 640 },
    IsCHW = true
};
```

## 批量转换多个模型

**PowerShell:**
```powershell
.\scripts\convert_paddle_models.ps1 `
  -InputDir "E:\models\paddle_models" `
  -OutputDir "E:\models\onnx_models"
```

**Python:**
```bash
python scripts/paddle2onnx_converter.py \
  --batch \
  --input_dir E:\models\paddle_models \
  --output_dir E:\models\onnx_models
```

## 常见问题

### 问：转换后的模型在哪里？

答：默认使用缓存，模型在：`%TEMP%\paddle2onnx_cache\`

### 问：如何清理缓存？

```csharp
var converter = new Paddle2OnnxConverter();
converter.ClearCache();
```

### 问：支持哪些 PaddlePaddle 模型？

答：大多数 PaddlePaddle 模型都支持，包括：
- PP-OCR (检测、识别、分类)
- PaddleClas (分类)
- PaddleDetection (目标检测)
- PaddleSeg (分割)
- 自定义模型（如果算子兼容）

### 问：转换失败怎么办？

1. 检查 paddle2onnx 是否安装：`pip show paddle2onnx`
2. 尝试不同的 opset 版本：`--opset_version 13`
3. 查看错误日志
4. 参考完整文档：`docs/PADDLE_OPENVINO_GUIDE.md`

## 完整文档

- **完整指南**: [docs/PADDLE_OPENVINO_GUIDE.md](PADDLE_OPENVINO_GUIDE.md)
- **测试指南**: [docs/PADDLE_TESTING_GUIDE.md](PADDLE_TESTING_GUIDE.md)
- **示例代码**: `dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs`
- **Rust 示例**: `cudars-ffi/examples/paddle_to_openvino.rs`

## 下一步

1. 阅读完整指南了解更多功能
2. 尝试不同的 PaddlePaddle 模型
3. 优化推理性能
4. 集成到你的应用中

---

**快速链接:**
- [完整使用指南](PADDLE_OPENVINO_GUIDE.md)
- [测试指南](PADDLE_TESTING_GUIDE.md)
- [主 README](../README.md)

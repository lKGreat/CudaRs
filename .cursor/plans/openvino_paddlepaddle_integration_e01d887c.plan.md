---
name: OpenVINO PaddlePaddle Integration
overview: 实现 OpenVINO 加载 PaddlePaddle 模型的完整解决方案，包括模型转换工具、预处理配置、完整示例和文档。
todos:
  - id: paddle2onnx-helper
    content: 创建 Rust 转换辅助模块（paddle2onnx_helper.rs）
    status: completed
  - id: csharp-converter
    content: 实现 C# 转换器和配置解析器
    status: completed
  - id: conversion-scripts
    content: 编写 PowerShell 和 Python 转换脚本
    status: completed
  - id: csharp-examples
    content: 创建完整的 C# 示例代码
    status: completed
  - id: rust-examples
    content: 创建 Rust 示例代码
    status: completed
  - id: documentation
    content: 编写完整的使用指南和文档
    status: completed
  - id: update-config
    content: 更新模型配置结构支持 PaddlePaddle
    status: completed
  - id: testing
    content: 测试各种 PaddlePaddle 模型的转换和推理
    status: completed
isProject: false
---

# OpenVINO 加载 PaddlePaddle 模型完整方案

## 背景分析

您的 PP-OCRv5 模型文件（`inference.json` + `inference.pdiparams` + `inference.yml`）使用的是 PaddlePaddle 3.0 的新 JSON 格式。OpenVINO 目前只支持旧的 `.pdmodel` 格式，因此需要通过转换来实现。

## 实现方案

### 方案选择

**推荐方案：Paddle2ONNX 转换流程**

- PaddlePaddle 模型 → ONNX → OpenVINO
- 优点：成熟稳定，社区支持好，无需训练权重
- 缺点：需要额外的转换步骤

**备选方案：直接使用 PaddlePaddle 推理引擎**

- 项目已有 PaddleOCR 集成 ([cudars-ffi/src/paddleocr.rs](cudars-ffi/src/paddleocr.rs))
- 可以直接加载原生模型，无需转换

## 实现内容

### 1. 模型转换工具集成

**文件：`cudars-ffi/src/paddle2onnx_helper.rs`**（新建）

- 提供 Paddle2ONNX 转换的辅助函数
- 自动检测模型文件格式（.json vs .pdmodel）
- 调用 paddle2onnx 命令行工具或 Python API

**文件：`dotnet/CudaRS/Paddle/Paddle2OnnxConverter.cs`**（新建）

- C# 包装器，调用转换工具
- 提供友好的 API 接口
- 缓存转换后的 ONNX 模型

### 2. 模型配置和预处理支持

**扩展：[cudars-ffi/src/sdk/openvino_model_config.rs](cudars-ffi/src/sdk/openvino_model_config.rs)**

```rust
pub struct OpenVinoModelConfig {
    pub model_path: String,
    pub auto_convert_paddle: bool,  // 新增：自动转换 PaddlePaddle 模型
    pub paddle_model_type: Option<String>,  // 新增：模型类型提示
}
```

**新建：`dotnet/CudaRS/Paddle/PaddlePreprocessConfig.cs`**

- 支持 PaddlePaddle 模型的预处理配置
- 从 `inference.yml` 读取配置参数
- 提供标准化的预处理接口（归一化、resize 等）

### 3. 完整示例代码

**Rust 示例：`cudars-ffi/examples/paddle_to_openvino.rs`**（新建）

```rust
// 示例：加载 PaddlePaddle 模型并用 OpenVINO 推理
fn main() {
    // 1. 转换模型（如果需要）
    let onnx_path = convert_paddle_to_onnx("inference.json", "model.onnx")?;
    
    // 2. 加载到 OpenVINO
    let config = CudaRsOvConfig { ... };
    let model = cudars_ov_load(&onnx_path, &config)?;
    
    // 3. 推理
    // ...
}
```

**C# 示例：`dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs`**（新建）

- 完整的端到端示例
- 从 PaddlePaddle 模型加载到 OpenVINO 推理
- 包含预处理和后处理
- 支持 OCR、检测、分类等常见任务

```csharp
public static void Run()
{
    // 1. 配置模型路径
    var paddleModelDir = @"E:\models\PP-OCRv5_mobile_det_infer";
    
    // 2. 自动转换或使用缓存
    var converter = new Paddle2OnnxConverter();
    var onnxPath = converter.ConvertOrUseCache(
        Path.Combine(paddleModelDir, "inference.json"),
        Path.Combine(paddleModelDir, "inference.pdiparams")
    );
    
    // 3. 加载配置
    var preprocessConfig = PaddlePreprocessConfig.FromYaml(
        Path.Combine(paddleModelDir, "inference.yml")
    );
    
    // 4. 创建 OpenVINO 模型
    var modelConfig = new OpenVinoModelConfig { ModelPath = onnxPath };
    using var model = new OpenVinoModel("paddle_det", modelConfig);
    using var pipeline = model.CreatePipeline("CPU");
    
    // 5. 推理
    var image = LoadImage("test.jpg");
    var preprocessed = preprocessConfig.Preprocess(image);
    var outputs = pipeline.Run(preprocessed.Data, preprocessed.Shape);
    
    // 6. 后处理
    var results = PostProcess(outputs, preprocessConfig);
    Console.WriteLine($"检测到 {results.Length} 个目标");
}
```

### 4. 转换脚本和工具

**PowerShell 脚本：`scripts/convert_paddle_models.ps1`**（新建）

```powershell
# 批量转换 PaddlePaddle 模型为 ONNX
param(
    [string]$InputDir,
    [string]$OutputDir
)

# 检查 paddle2onnx 安装
# 遍历模型文件
# 执行转换
# 验证转换结果
```

**Python 脚本：`scripts/paddle2onnx_converter.py`**（新建）

```python
import paddle2onnx
import argparse

def convert_paddle_to_onnx(json_path, params_path, output_path):
    """转换 PaddlePaddle 模型为 ONNX"""
    # 读取模型
    # 执行转换
    # 保存 ONNX
    # 验证模型
```

### 5. 文档和指南

**文档：`docs/PADDLE_OPENVINO_GUIDE.md`**（新建）

内容大纲：

- PaddlePaddle 模型格式说明（.json vs .pdmodel）
- Paddle2ONNX 安装和配置
- 模型转换完整流程
- OpenVINO 加载和推理
- 预处理配置说明
- 常见问题和解决方案
- 性能优化建议

**更新：[README.md](README.md)**

- 添加 PaddlePaddle 模型支持章节
- 链接到详细指南
- 提供快速开始示例

### 6. 配置助手工具

**新建：`dotnet/CudaRS/Paddle/PaddleYamlParser.cs`**

```csharp
public class PaddleYamlParser
{
    // 解析 inference.yml 配置文件
    // 提取预处理参数
    // 转换为 OpenVINO 可用的配置
}
```

## 技术要点

### Paddle2ONNX 转换

```bash
# 安装
pip install paddle2onnx onnx

# 转换命令（针对 .json 格式）
paddle2onnx \
  --model_dir ./PP-OCRv5_mobile_det_infer \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file model.onnx \
  --opset_version 11
```

### 预处理配置解析

inference.yml 中通常包含：

- `mean`: 归一化均值
- `std`: 归一化标准差
- `scale`: 缩放因子
- `image_shape`: 输入尺寸
- `transforms`: 预处理步骤

### OpenVINO 性能优化

```csharp
var pipelineConfig = new OpenVinoPipelineConfig
{
    OpenVinoDevice = "CPU",
    OpenVinoPerformanceMode = "throughput",
    OpenVinoNumStreams = 4,  // 多流并行
    OpenVinoCacheDir = "./ov_cache",  // 模型缓存
    OpenVinoEnableMmap = true  // 内存映射加载
};
```

## 文件清单

### 新增文件

1. `cudars-ffi/src/paddle2onnx_helper.rs` - Rust 转换辅助
2. `cudars-ffi/examples/paddle_to_openvino.rs` - Rust 示例
3. `dotnet/CudaRS/Paddle/Paddle2OnnxConverter.cs` - C# 转换器
4. `dotnet/CudaRS/Paddle/PaddlePreprocessConfig.cs` - 预处理配置
5. `dotnet/CudaRS/Paddle/PaddleYamlParser.cs` - YAML 解析器
6. `dotnet/CudaRS.Examples/Tests/CasePaddleOpenVinoTest.cs` - C# 示例
7. `scripts/convert_paddle_models.ps1` - PowerShell 转换脚本
8. `scripts/paddle2onnx_converter.py` - Python 转换脚本
9. `docs/PADDLE_OPENVINO_GUIDE.md` - 完整指南

### 修改文件

1. [cudars-ffi/src/sdk/openvino_model_config.rs](cudars-ffi/src/sdk/openvino_model_config.rs) - 添加 PaddlePaddle 支持字段
2. [dotnet/CudaRS/OpenVino/OpenVinoModelConfig.cs](dotnet/CudaRS/OpenVino/OpenVinoModelConfig.cs) - 同步配置字段
3. [README.md](README.md) - 添加 PaddlePaddle 支持说明

## 使用流程

### 快速开始

```csharp
// 一行代码搞定转换和加载
var model = OpenVinoModel.FromPaddlePaddle(
    "paddle_det",
    @"E:\models\PP-OCRv5_mobile_det_infer",
    autoConvert: true
);
```

### 手动流程

```bash
# 1. 转换模型
paddle2onnx --model_dir ./model_dir --save_file model.onnx

# 2. C# 代码加载
var config = new OpenVinoModelConfig { ModelPath = "model.onnx" };
var model = new OpenVinoModel("det", config);
```

## 依赖项

### Python 环境

- `paddle2onnx >= 1.2.0`
- `onnx >= 1.15.0`
- `onnxruntime >= 1.16.0`（用于验证）

### .NET NuGet 包

- `YamlDotNet` - 解析 inference.yml（如果需要）

## 测试计划

1. 测试 OCR 检测模型转换和推理
2. 测试 OCR 识别模型转换和推理
3. 测试分类模型
4. 测试检测模型
5. 性能基准测试（vs PaddlePaddle 原生）

## 注意事项

1. **模型兼容性**：不是所有 PaddlePaddle 算子都能转换为 ONNX，需要测试验证
2. **精度损失**：转换可能导致轻微精度损失，需要对比测试
3. **性能对比**：OpenVINO 在某些硬件上可能比 PaddlePaddle 更快，但需实测
4. **缓存机制**：转换后的 ONNX 模型应该缓存，避免重复转换


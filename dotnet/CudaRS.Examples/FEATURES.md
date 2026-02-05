# CudaRS 新功能示例

本文档展示 CudaRS SDK 的最新优化功能。

## 1. 图片画框功能

### 基本用法

```csharp
using CudaRS.Yolo;

// 创建检测 pipeline
var pipeline = CudaRsFluent.Create()
    .Pipeline()
    .ForYolo("model.onnx", cfg => { /* ... */ })
    .AsOpenVino()
    .BuildYoloFluent();

// 方式1: 纯检测数据
var detections = pipeline.Run(imageBytes).AsDetections();

// 方式2: 带框图片
var annotated = pipeline.Run(imageBytes).AsAnnotatedImage(
    new AnnotationOptions
    {
        ShowLabel = true,
        ShowConfidence = true,
        BoxThickness = 3f,
        FontSize = 14f
    },
    ImageFormat.Jpeg);

// 保存结果
File.WriteAllBytes("output.jpg", annotated.ImageBytes);

// 方式3: 两者都要
var combined = pipeline.Run(imageBytes).AsCombined();
Console.WriteLine($"检测到 {combined.Inference.Detections.Count} 个对象");
File.WriteAllBytes("output.png", combined.AnnotatedImage.ImageBytes);
```

### 自定义颜色

```csharp
var options = new AnnotationOptions
{
    ClassColors = new Dictionary<int, Color>
    {
        [0] = Color.Red,      // 类别0用红色
        [1] = Color.Blue,     // 类别1用蓝色
        [2] = Color.Green,    // 类别2用绿色
    }
};

var result = pipeline.Run(imageBytes)
    .AsAnnotatedImage(options);
```

## 2. 灵活的结果返回模式

### As 函数模式

通过 As 函数链式调用，灵活选择返回结果类型：

```csharp
// 仅获取检测数据（最快，无图像处理开销）
var data = pipeline.Run(imageBytes).AsDetections();

// 仅获取带框图片（用于可视化）
var image = pipeline.Run(imageBytes).AsAnnotatedImage();

// 同时获取数据和图片
var both = pipeline.Run(imageBytes).AsCombined();
```

### 性能对比

| 模式 | 用途 | 性能 |
|------|------|------|
| AsDetections() | 纯数据处理 | 最快 |
| AsAnnotatedImage() | 仅可视化 | 中等 |
| AsCombined() | 完整结果 | 较慢 |

## 3. 详细的错误报告

### 增强的错误信息

新增错误类型：
- `MissingDependency` - 缺少依赖
- `DllNotFound` - DLL文件未找到
- `ModelLoadFailed` - 模型加载失败
- `ConfigInvalid` - 配置无效

### 结构化错误详情

```csharp
try
{
    var pipeline = /* ... */;
}
catch (SdkException ex)
{
    Console.WriteLine($"错误码: {ex.ErrorCode}");
    Console.WriteLine($"消息: {ex.Message}");
    
    // 新增字段
    if (ex.MissingFile != null)
        Console.WriteLine($"缺失文件: {ex.MissingFile}");
    
    if (ex.SearchedPaths != null)
    {
        Console.WriteLine("已搜索路径:");
        foreach (var path in ex.SearchedPaths)
            Console.WriteLine($"  - {path}");
    }
    
    if (ex.Suggestion != null)
        Console.WriteLine($"解决建议: {ex.Suggestion}");
}
```

### 编译时依赖检测

Rust 项目的 `build.rs` 现在会详细报告缺失的依赖：

```
依赖 'OpenVINO' 未找到
至少一个文件: ["openvino.lib", "openvino_c.lib"]
已搜索路径:
  C:\openvino
  E:\codeding\AI\cudars\openvino_env\Lib\site-packages\openvino\libs
解决方案:
1. 设置环境变量 OPENVINO_ROOT 或 OPENVINO_LIB 指向安装目录
2. 或将依赖安装到默认位置
3. 详细信息请参阅项目文档
```

## 4. 使用技巧

### 批量画框处理

```csharp
var images = Directory.GetFiles("input", "*.jpg");
var outputDir = "output";
Directory.CreateDirectory(outputDir);

foreach (var imagePath in images)
{
    var bytes = File.ReadAllBytes(imagePath);
    var result = pipeline.Run(bytes).AsAnnotatedImage();
    
    var outputPath = Path.Combine(outputDir, 
        $"annotated_{Path.GetFileName(imagePath)}");
    File.WriteAllBytes(outputPath, result.ImageBytes);
}
```

### 条件画框

```csharp
var wrapper = pipeline.Run(imageBytes);
var detections = wrapper.AsDetections();

if (detections.Detections.Count > 0)
{
    // 有检测结果才画框
    var annotated = wrapper.AsAnnotatedImage();
    File.WriteAllBytes("output.jpg", annotated.ImageBytes);
}
else
{
    Console.WriteLine("未检测到对象");
}
```

### 高置信度过滤

```csharp
var result = pipeline.Run(imageBytes).AsDetections();
var highConfDetections = result.Detections
    .Where(d => d.Confidence > 0.8f)
    .ToList();

// 使用 ImageAnnotator 手动画框
using var image = Image.Load<Rgb24>(imageBytes.ToArray());
ImageAnnotator.DrawBoxes(image, highConfDetections, new AnnotationOptions
{
    ShowConfidence = true,
    BoxThickness = 4f
});

image.SaveAsJpeg("high_conf_output.jpg");
```

## 5. 调试和诊断

### 启用诊断输出

设置环境变量：
```bash
# Windows
set CUDARS_DIAG=1

# Linux/Mac
export CUDARS_DIAG=1
```

### 查看依赖搜索路径

构建时会显示详细的依赖搜索过程：
```bash
cargo build --features openvino
```

### 自定义 FFI 库路径

```bash
# 指定 FFI 库位置
set CUDARS_FFI_PATH=E:\path\to\cudars_ffi.dll

# 或指定目录
set CUDARS_FFI_DIR=E:\path\to\libs
```

## 6. 完整示例

参见 `Program.cs` 中的 `RunAnnotationDemo()` 方法和 `ErrorHandlingExample.cs`。

运行示例：
```bash
cd dotnet/CudaRS.Examples
dotnet run
```

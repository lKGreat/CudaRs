# CudaRS C# SDK OpenVINO集成方案

## 一、当前C# SDK架构分析

### 1.1 现有层次结构

```
CudaRS.Native (P/Invoke层)
    ↓
CudaRS.Core (SafeHandle + 异常处理)
    ↓
CudaRS (高级API: OpenVino, Ocr)
    ↓
CudaRS.Yolo (Fluent Builder + YOLO专用)
    ↓
CudaRS.Examples (示例和测试)
```

### 1.2 现有OpenVINO C# API

#### 基础API (CudaRS.OpenVino)
- **OpenVinoModel** - 模型管理
- **OpenVinoPipeline** - 通用张量推理
- **OpenVinoModelConfig** - 模型配置
- **OpenVinoPipelineConfig** - 推理配置
- **OpenVinoTensorOutput** - 输出结构

#### 原生API (CudaRS.OpenVino)
- **OpenVinoNativeModel** - 直接FFI调用
- **OpenVinoAsyncQueue** - 异步队列

#### Fluent API (CudaRS.Yolo)
- **FluentPipelineBuilder.AsOpenVino()** - YOLO OpenVINO后端
- **FluentPipelineBuilder.ForTensor()** - 通用张量

---

## 二、新功能C# API设计

### 2.1 动态形状支持

#### 新增类型
```csharp
namespace CudaRS.OpenVino;

/// <summary>
/// 表示部分维度(支持动态)
/// </summary>
public readonly struct PartialDimension
{
    public bool IsStatic { get; }
    public long Value { get; }  // -1表示动态
    
    public static PartialDimension Static(long value) => new(true, value);
    public static PartialDimension Dynamic => new(false, -1);
}

/// <summary>
/// 部分形状(支持动态维度)
/// </summary>
public sealed class PartialShape
{
    public PartialDimension[] Dimensions { get; set; }
    
    public static PartialShape Create(params long[] dims);
    public static PartialShape CreateDynamic(int rank);
    public bool IsDynamic { get; }
}
```

#### 扩展OpenVinoModel
```csharp
namespace CudaRS.OpenVino;

public sealed class OpenVinoModel
{
    // 现有方法...
    
    /// <summary>
    /// 获取模型输入元信息
    /// </summary>
    public ModelTensorInfo[] GetInputs();
    
    /// <summary>
    /// 获取模型输出元信息
    /// </summary>
    public ModelTensorInfo[] GetOutputs();
    
    /// <summary>
    /// 重塑模型输入形状(支持动态)
    /// </summary>
    public void Reshape(params PartialShape[] inputShapes);
}

/// <summary>
/// 张量元信息
/// </summary>
public sealed class ModelTensorInfo
{
    public string Name { get; set; }
    public PartialShape Shape { get; set; }
    public TensorElementType ElementType { get; set; }
}

public enum TensorElementType
{
    Undefined = 0,
    F32 = 4,
    F16 = 3,
    I32 = 9,
    I64 = 10,
    U8 = 16,
}
```

### 2.2 预处理API

```csharp
namespace CudaRS.OpenVino.Preprocessing;

/// <summary>
/// 预处理构建器
/// </summary>
public sealed class PreprocessBuilder
{
    public PreprocessBuilder Input(string? name = null);
    public PreprocessBuilder Output(string? name = null);
    
    // 输入张量配置
    public PreprocessBuilder TensorFormat(TensorElementType type);
    public PreprocessBuilder TensorLayout(string layout);  // "NHWC", "NCHW"
    
    // 模型配置
    public PreprocessBuilder ModelLayout(string layout);
    public PreprocessBuilder ModelFormat(TensorElementType type);
    
    // 预处理步骤
    public PreprocessBuilder Resize(ResizeAlgorithm algorithm);
    public PreprocessBuilder ConvertElementType(TensorElementType targetType);
    public PreprocessBuilder ConvertLayout(string targetLayout);
    public PreprocessBuilder Mean(params float[] values);
    public PreprocessBuilder Scale(params float[] values);
    
    // 构建新模型
    public OpenVinoModel Build(OpenVinoModel originalModel);
}

public enum ResizeAlgorithm
{
    Linear,
    Nearest,
    Cubic,
}

// 使用示例
var preprocessed = new PreprocessBuilder()
    .Input()
        .TensorFormat(TensorElementType.U8)
        .TensorLayout("NHWC")
        .ModelLayout("NCHW")
        .ModelFormat(TensorElementType.F32)
        .Resize(ResizeAlgorithm.Linear)
        .ConvertElementType(TensorElementType.F32)
        .Mean(0f, 0f, 0f)
        .Scale(255f, 255f, 255f)
    .Build(originalModel);
```

### 2.3 批处理推理

#### 扩展OpenVinoPipeline
```csharp
namespace CudaRS.OpenVino;

public sealed class OpenVinoPipeline
{
    // 现有方法...
    
    /// <summary>
    /// 批量推理
    /// </summary>
    public OpenVinoTensorOutput[][] RunBatch(
        ReadOnlyMemory<float>[] inputs,
        ReadOnlyMemory<long> shape);
    
    /// <summary>
    /// 异步批量推理
    /// </summary>
    public Task<OpenVinoTensorOutput[][]> RunBatchAsync(
        ReadOnlyMemory<float>[] inputs,
        ReadOnlyMemory<long> shape,
        CancellationToken cancellationToken = default);
}
```

#### YOLO批处理扩展
```csharp
namespace CudaRS.Yolo;

public interface IFluentImagePipeline<TResult>
{
    // 现有方法...
    
    /// <summary>
    /// 批量推理多张图片
    /// </summary>
    TResult[] RunBatch(params byte[][] images);
    
    /// <summary>
    /// 异步批量推理
    /// </summary>
    Task<TResult[]> RunBatchAsync(
        byte[][] images,
        CancellationToken cancellationToken = default);
}

// 使用示例
var images = new[] {
    File.ReadAllBytes("img1.jpg"),
    File.ReadAllBytes("img2.jpg"),
    File.ReadAllBytes("img3.jpg"),
};

var results = pipeline.RunBatch(images);
foreach (var result in results)
{
    Console.WriteLine($"Detected: {result.Detections.Count}");
}
```

### 2.4 性能分析器

```csharp
namespace CudaRS.OpenVino.Profiling;

/// <summary>
/// 推理性能分析器
/// </summary>
public sealed class InferenceProfiler
{
    public bool Enabled { get; set; }
    
    public ProfilingResult Profile(Action<OpenVinoPipeline> inferAction);
}

/// <summary>
/// 性能分析结果
/// </summary>
public sealed class ProfilingResult
{
    public TimeSpan TotalTime { get; set; }
    public TimeSpan PreprocessTime { get; set; }
    public TimeSpan InferenceTime { get; set; }
    public TimeSpan PostprocessTime { get; set; }
    
    // 层级时间(如果启用)
    public Dictionary<string, TimeSpan>? LayerTimes { get; set; }
    
    public void Print();
}

// 使用示例
var profiler = new InferenceProfiler { Enabled = true };
var result = profiler.Profile(p => p.Run(input, shape));

result.Print();
// 输出:
// Total: 15.2ms
// - Preprocess: 2.1ms
// - Inference: 12.5ms
// - Postprocess: 0.6ms
```

### 2.5 INT8量化支持

```csharp
namespace CudaRS.OpenVino;

/// <summary>
/// 模型精度配置
/// </summary>
public sealed class PrecisionConfig
{
    public ModelPrecision Precision { get; set; } = ModelPrecision.Auto;
    public bool EnableInt8Calibration { get; set; }
    public string? CalibrationDatasetPath { get; set; }
}

public enum ModelPrecision
{
    Auto,      // 自动检测
    FP32,
    FP16,
    INT8,
    Mixed,     // 混合精度
}

// 扩展OpenVinoModelConfig
public sealed class OpenVinoModelConfig
{
    public string ModelPath { get; set; }
    public PrecisionConfig? Precision { get; set; }
}

// 使用示例
var config = new OpenVinoModelConfig
{
    ModelPath = "model.onnx",
    Precision = new PrecisionConfig
    {
        Precision = ModelPrecision.INT8,
    }
};
```

### 2.6 高级配置扩展

```csharp
namespace CudaRS.OpenVino;

/// <summary>
/// 高级性能配置
/// </summary>
public sealed class AdvancedConfig
{
    // 亲和性设置
    public CpuAffinityMode? CpuAffinity { get; set; }
    public int[]? CpuThreadsPerStream { get; set; }
    
    // GPU配置
    public GpuExecutionMode? GpuMode { get; set; }
    public int? GpuQueueThrottle { get; set; }
    
    // 优化选项
    public bool? EnableDynamicShapes { get; set; }
    public bool? EnableModelCaching { get; set; }
    public bool? EnableCpuPinning { get; set; }
}

public enum CpuAffinityMode
{
    None,
    NUMA,
    HybridAware,
}

public enum GpuExecutionMode
{
    Auto,
    Queue,
    Stream,
}
```

---

## 三、C# Native层扩展

### 3.1 新增P/Invoke方法

```csharp
// dotnet/CudaRS.Native/OpenVinoNative.cs

public static unsafe partial class SdkNative
{
    // ===== 动态形状 =====
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_input_info")]
    public static partial CudaRsResult OpenVinoGetInputInfo(
        ulong handle,
        int index,
        long* outShape,
        int* outShapeLen,
        int maxShapeLen);
    
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_reshape")]
    public static partial CudaRsResult OpenVinoReshape(
        ulong handle,
        CudaRsOvPartialShape* inputShapes,
        int numInputs);
    
    // ===== 批处理 =====
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_run_batch")]
    public static partial CudaRsResult OpenVinoRunBatch(
        ulong handle,
        float** inputs,
        ulong* inputLengths,
        int batchSize,
        long* shape,
        ulong shapeLen,
        out CudaRsOvTensor** tensors,
        out ulong* countsPerBatch,
        out ulong totalOutputs);
    
    // ===== 性能分析 =====
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_perf_counts")]
    public static partial CudaRsResult OpenVinoGetPerfCounts(
        ulong handle,
        out CudaRsOvPerfData* perfData);
    
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_free_perf_data")]
    public static partial void OpenVinoFreePerfData(CudaRsOvPerfData* perfData);
}
```

### 3.2 新增结构体

```csharp
// dotnet/CudaRS.Native/OpenVinoNativeTypes.cs

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvPartialDim
{
    public byte IsStatic;
    public long Value;
}

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvPartialShape
{
    public int Rank;
    public CudaRsOvPartialDim* Dims;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct CudaRsOvPerfData
{
    public byte* JsonDataPtr;
    public nuint JsonDataLen;
}
```

---

## 四、CudaRS.Examples测试方案

### 4.1 新增测试场景

#### 测试文件结构
```
CudaRS.Examples/
├── Config.cs (扩展配置)
├── Program.cs (主入口)
├── Tests/
│   ├── OpenVinoBasicTests.cs      ✨新增
│   ├── OpenVinoDynamicShapeTests.cs ✨新增
│   ├── OpenVinoBatchTests.cs      ✨新增
│   ├── OpenVinoProfilingTests.cs  ✨新增
│   ├── OpenVinoInt8Tests.cs       ✨新增
│   └── OpenVinoPreprocessTests.cs ✨新增
├── Benchmarks/
│   ├── ThroughputBenchmark.cs     ✨新增
│   └── LatencyBenchmark.cs        ✨新增
└── Utils/
    └── TestDataGenerator.cs        ✨新增
```

### 4.2 OpenVinoBasicTests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoBasicTests
{
    /// <summary>
    /// 测试1: 基础推理能力
    /// </summary>
    public static void TestBasicInference()
    {
        Console.WriteLine("\n[Test 1] Basic Inference");
        
        var model = new OpenVinoModel("basic", new OpenVinoModelConfig
        {
            ModelPath = Config.TestModelPath,
        });
        
        var pipeline = model.CreatePipeline("default", new OpenVinoPipelineConfig
        {
            OpenVinoDevice = "cpu",
        });
        
        var input = new float[1 * 3 * 640 * 640];
        var shape = new long[] { 1, 3, 640, 640 };
        
        var sw = Stopwatch.StartNew();
        var outputs = pipeline.Run(input, shape);
        sw.Stop();
        
        Console.WriteLine($"✓ Inference completed: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"✓ Output count: {outputs.Length}");
        
        Assert(outputs.Length > 0, "Should have outputs");
    }
    
    /// <summary>
    /// 测试2: 多设备支持
    /// </summary>
    public static void TestMultipleDevices()
    {
        Console.WriteLine("\n[Test 2] Multiple Devices");
        
        var devices = new[] { "cpu", "gpu", "auto" };
        
        foreach (var device in devices)
        {
            try
            {
                var pipeline = CreatePipeline(device);
                var result = RunSampleInference(pipeline);
                Console.WriteLine($"✓ Device '{device}': {result.ElapsedMs}ms");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Device '{device}': {ex.Message}");
            }
        }
    }
    
    /// <summary>
    /// 测试3: 异步推理
    /// </summary>
    public static async Task TestAsyncInference()
    {
        Console.WriteLine("\n[Test 3] Async Inference");
        
        var native = new OpenVinoNativeModel(Config.TestModelPath, 
            new OpenVinoNativeConfig
            {
                Device = "cpu",
                NumRequests = 4,
            });
        
        var queue = native.CreateAsyncQueue();
        
        var tasks = Enumerable.Range(0, 10)
            .Select(async i =>
            {
                var input = GenerateRandomInput();
                var shape = new long[] { 1, 3, 640, 640 };
                
                var requestId = queue.Submit(input, shape);
                var outputs = queue.Wait(requestId);
                
                return (i, outputs);
            })
            .ToArray();
        
        var results = await Task.WhenAll(tasks);
        
        Console.WriteLine($"✓ Completed {results.Length} async requests");
    }
    
    private static void Assert(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException($"Assertion failed: {message}");
    }
}
```

### 4.3 OpenVinoDynamicShapeTests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoDynamicShapeTests
{
    /// <summary>
    /// 测试4: 动态形状推理
    /// </summary>
    public static void TestDynamicShapes()
    {
        Console.WriteLine("\n[Test 4] Dynamic Shapes");
        
        var model = LoadModel();
        
        // 查询原始形状
        var inputs = model.GetInputs();
        Console.WriteLine($"Original input shape: {FormatShape(inputs[0].Shape)}");
        
        // 测试不同输入尺寸
        var testSizes = new[] { 320, 640, 960, 1280 };
        
        foreach (var size in testSizes)
        {
            // 重塑模型
            model.Reshape(PartialShape.Create(1, 3, size, size));
            
            var pipeline = model.CreatePipeline("dynamic", new OpenVinoPipelineConfig
            {
                OpenVinoDevice = "cpu",
            });
            
            var input = new float[1 * 3 * size * size];
            var shape = new long[] { 1, 3, size, size };
            
            var sw = Stopwatch.StartNew();
            var outputs = pipeline.Run(input, shape);
            sw.Stop();
            
            Console.WriteLine($"✓ Size {size}x{size}: {sw.ElapsedMilliseconds}ms");
        }
    }
    
    /// <summary>
    /// 测试5: 批大小动态
    /// </summary>
    public static void TestDynamicBatchSize()
    {
        Console.WriteLine("\n[Test 5] Dynamic Batch Size");
        
        var model = LoadModel();
        
        // 设置batch维度为动态
        model.Reshape(PartialShape.CreateDynamic(4)); // [?, 3, 640, 640]
        
        var batchSizes = new[] { 1, 2, 4, 8 };
        
        foreach (var batchSize in batchSizes)
        {
            var input = new float[batchSize * 3 * 640 * 640];
            var shape = new long[] { batchSize, 3, 640, 640 };
            
            var pipeline = model.CreatePipeline($"batch_{batchSize}", 
                new OpenVinoPipelineConfig { OpenVinoDevice = "cpu" });
            
            var outputs = pipeline.Run(input, shape);
            Console.WriteLine($"✓ Batch {batchSize}: {outputs.Length} outputs");
        }
    }
}
```

### 4.4 OpenVinoBatchTests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoBatchTests
{
    /// <summary>
    /// 测试6: 批处理推理
    /// </summary>
    public static void TestBatchInference()
    {
        Console.WriteLine("\n[Test 6] Batch Inference");
        
        var images = new[]
        {
            File.ReadAllBytes(Config.TestImage1),
            File.ReadAllBytes(Config.TestImage2),
            File.ReadAllBytes(Config.TestImage3),
            File.ReadAllBytes(Config.TestImage4),
        };
        
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(Config.TestYoloModel, cfg =>
            {
                cfg.InputWidth = 640;
                cfg.InputHeight = 640;
            })
            .AsOpenVino()
            .WithThroughput(t =>
            {
                t.Enable = true;
                t.BatchSize = 4;
            })
            .BuildYolo();
        
        var sw = Stopwatch.StartNew();
        var results = pipeline.RunBatch(images);
        sw.Stop();
        
        Console.WriteLine($"✓ Batch inference: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"✓ Throughput: {images.Length * 1000.0 / sw.ElapsedMilliseconds:F2} imgs/sec");
        
        for (int i = 0; i < results.Length; i++)
        {
            Console.WriteLine($"  Image {i}: {results[i].Detections.Count} detections");
        }
    }
    
    /// <summary>
    /// 测试7: 批处理性能对比
    /// </summary>
    public static void CompareBatchVsSingle()
    {
        Console.WriteLine("\n[Test 7] Batch vs Single Performance");
        
        var images = LoadTestImages(8);
        
        // 单张推理
        var singlePipeline = CreatePipeline(batchSize: 1);
        var singleTime = MeasureSingleInference(singlePipeline, images);
        
        // 批量推理
        var batchPipeline = CreatePipeline(batchSize: 8);
        var batchTime = MeasureBatchInference(batchPipeline, images);
        
        Console.WriteLine($"Single inference: {singleTime}ms ({images.Length * 1000.0 / singleTime:F2} imgs/sec)");
        Console.WriteLine($"Batch inference:  {batchTime}ms ({images.Length * 1000.0 / batchTime:F2} imgs/sec)");
        Console.WriteLine($"Speedup: {singleTime / batchTime:F2}x");
    }
}
```

### 4.5 OpenVinoProfilingTests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoProfilingTests
{
    /// <summary>
    /// 测试8: 性能分析
    /// </summary>
    public static void TestProfiling()
    {
        Console.WriteLine("\n[Test 8] Performance Profiling");
        
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(Config.TestYoloModel, cfg => { })
            .AsOpenVino()
            .BuildYolo();
        
        var profiler = new InferenceProfiler { Enabled = true };
        
        var imageBytes = File.ReadAllBytes(Config.TestImage1);
        
        // 预热
        for (int i = 0; i < 5; i++)
            pipeline.Run(imageBytes);
        
        // 性能分析
        var result = profiler.Profile(p =>
        {
            for (int i = 0; i < 100; i++)
                pipeline.Run(imageBytes);
        });
        
        result.Print();
    }
    
    /// <summary>
    /// 测试9: 层级性能分析
    /// </summary>
    public static void TestLayerProfiling()
    {
        Console.WriteLine("\n[Test 9] Layer Profiling");
        
        var config = new OpenVinoPipelineConfig
        {
            OpenVinoDevice = "cpu",
            OpenVinoEnableProfiling = true,
        };
        
        var model = new OpenVinoModel("profiled", new OpenVinoModelConfig
        {
            ModelPath = Config.TestYoloModel,
        });
        
        var pipeline = model.CreatePipeline("default", config);
        
        // 推理
        var input = GenerateRandomInput();
        var shape = new long[] { 1, 3, 640, 640 };
        pipeline.Run(input, shape);
        
        // 获取层级性能
        var layerTimes = pipeline.GetLayerTimings();
        
        Console.WriteLine("Top 10 slowest layers:");
        foreach (var (layer, time) in layerTimes.OrderByDescending(x => x.Value).Take(10))
        {
            Console.WriteLine($"  {layer}: {time.TotalMilliseconds:F3}ms");
        }
    }
}
```

### 4.6 OpenVinoInt8Tests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoInt8Tests
{
    /// <summary>
    /// 测试10: INT8量化模型
    /// </summary>
    public static void TestInt8Inference()
    {
        Console.WriteLine("\n[Test 10] INT8 Quantized Model");
        
        if (!File.Exists(Config.TestYoloInt8Model))
        {
            Console.WriteLine("⚠ INT8 model not found, skipping");
            return;
        }
        
        // FP32模型
        var fp32Pipeline = CreatePipeline(Config.TestYoloModel);
        var fp32Time = MeasureInferenceTime(fp32Pipeline);
        
        // INT8模型
        var int8Pipeline = CreatePipeline(Config.TestYoloInt8Model, new PrecisionConfig
        {
            Precision = ModelPrecision.INT8,
        });
        var int8Time = MeasureInferenceTime(int8Pipeline);
        
        Console.WriteLine($"FP32:  {fp32Time}ms");
        Console.WriteLine($"INT8:  {int8Time}ms");
        Console.WriteLine($"Speedup: {fp32Time / int8Time:F2}x");
        
        // 精度对比
        CompareAccuracy(fp32Pipeline, int8Pipeline);
    }
    
    private static void CompareAccuracy(
        IFluentImagePipeline<ModelInferenceResult> fp32,
        IFluentImagePipeline<ModelInferenceResult> int8)
    {
        var imageBytes = File.ReadAllBytes(Config.TestImage1);
        
        var fp32Result = fp32.Run(imageBytes);
        var int8Result = int8.Run(imageBytes);
        
        Console.WriteLine($"FP32 detections: {fp32Result.Detections.Count}");
        Console.WriteLine($"INT8 detections: {int8Result.Detections.Count}");
        
        // 计算IoU相似度
        var iou = CalculateIoU(fp32Result.Detections, int8Result.Detections);
        Console.WriteLine($"Average IoU: {iou:F3}");
    }
}
```

### 4.7 OpenVinoPreprocessTests.cs

```csharp
namespace CudaRS.Examples.Tests;

public static class OpenVinoPreprocessTests
{
    /// <summary>
    /// 测试11: GPU预处理
    /// </summary>
    public static void TestGpuPreprocessing()
    {
        Console.WriteLine("\n[Test 11] GPU Preprocessing");
        
        var originalModel = new OpenVinoModel("original", new OpenVinoModelConfig
        {
            ModelPath = Config.TestYoloModel,
        });
        
        // 使用预处理API
        var preprocessedModel = new PreprocessBuilder()
            .Input()
                .TensorFormat(TensorElementType.U8)
                .TensorLayout("NHWC")
                .ModelLayout("NCHW")
                .ModelFormat(TensorElementType.F32)
                .Resize(ResizeAlgorithm.Linear)
                .ConvertElementType(TensorElementType.F32)
                .Scale(255f, 255f, 255f)
            .Build(originalModel);
        
        // CPU预处理
        var cpuPipeline = originalModel.CreatePipeline("cpu", new OpenVinoPipelineConfig
        {
            OpenVinoDevice = "cpu",
        });
        
        // GPU预处理
        var gpuPipeline = preprocessedModel.CreatePipeline("gpu", new OpenVinoPipelineConfig
        {
            OpenVinoDevice = "gpu",
        });
        
        var imageBytes = File.ReadAllBytes(Config.TestImage1);
        
        // 测试CPU预处理
        var cpuTime = MeasureTime(() =>
        {
            var input = PreprocessOnCpu(imageBytes);
            cpuPipeline.Run(input, new long[] { 1, 3, 640, 640 });
        });
        
        // 测试GPU预处理
        var gpuTime = MeasureTime(() =>
        {
            var rawInput = DecodeImage(imageBytes);
            gpuPipeline.Run(rawInput, new long[] { 1, 3, 640, 640 });
        });
        
        Console.WriteLine($"CPU preprocess: {cpuTime}ms");
        Console.WriteLine($"GPU preprocess: {gpuTime}ms");
        Console.WriteLine($"Speedup: {cpuTime / gpuTime:F2}x");
    }
}
```

### 4.8 性能基准测试

```csharp
namespace CudaRS.Examples.Benchmarks;

public static class ThroughputBenchmark
{
    /// <summary>
    /// 吞吐量基准测试
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("\n=== Throughput Benchmark ===");
        
        var configs = new[]
        {
            ("CPU, 1 stream", new { Device = "cpu", Streams = 1, Requests = 1 }),
            ("CPU, 4 streams", new { Device = "cpu", Streams = 4, Requests = 4 }),
            ("CPU, AUTO", new { Device = "cpu", Streams = 0, Requests = 8 }),
            ("GPU, 1 stream", new { Device = "gpu", Streams = 1, Requests = 1 }),
            ("GPU, throughput", new { Device = "gpu", Streams = 0, Requests = 4 }),
        };
        
        var results = new List<(string Config, double Throughput)>();
        
        foreach (var (name, config) in configs)
        {
            try
            {
                var throughput = MeasureThroughput(config.Device, config.Streams, config.Requests);
                results.Add((name, throughput));
                Console.WriteLine($"{name}: {throughput:F2} imgs/sec");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{name}: Failed - {ex.Message}");
            }
        }
        
        // 输出排序结果
        Console.WriteLine("\n--- Ranked Results ---");
        foreach (var (config, throughput) in results.OrderByDescending(x => x.Throughput))
        {
            Console.WriteLine($"{throughput:F2} imgs/sec - {config}");
        }
    }
    
    private static double MeasureThroughput(string device, int streams, int requests)
    {
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(Config.TestYoloModel, cfg => { })
            .AsOpenVino()
            .WithThroughput(t =>
            {
                t.Enable = true;
                t.NumStreams = streams;
            })
            .BuildYolo();
        
        var imageBytes = File.ReadAllBytes(Config.TestImage1);
        
        // 预热
        for (int i = 0; i < 10; i++)
            pipeline.Run(imageBytes);
        
        // 测试
        var count = 100;
        var sw = Stopwatch.StartNew();
        
        for (int i = 0; i < count; i++)
            pipeline.Run(imageBytes);
        
        sw.Stop();
        
        return count * 1000.0 / sw.ElapsedMilliseconds;
    }
}

public static class LatencyBenchmark
{
    /// <summary>
    /// 延迟基准测试
    /// </summary>
    public static void Run()
    {
        Console.WriteLine("\n=== Latency Benchmark ===");
        
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(Config.TestYoloModel, cfg => { })
            .AsOpenVino()
            .BuildYolo();
        
        var imageBytes = File.ReadAllBytes(Config.TestImage1);
        
        // 预热
        for (int i = 0; i < 10; i++)
            pipeline.Run(imageBytes);
        
        // 测量延迟
        var latencies = new List<double>();
        
        for (int i = 0; i < 100; i++)
        {
            var sw = Stopwatch.StartNew();
            pipeline.Run(imageBytes);
            sw.Stop();
            latencies.Add(sw.Elapsed.TotalMilliseconds);
        }
        
        Console.WriteLine($"Avg latency: {latencies.Average():F2}ms");
        Console.WriteLine($"Min latency: {latencies.Min():F2}ms");
        Console.WriteLine($"Max latency: {latencies.Max():F2}ms");
        Console.WriteLine($"P50 latency: {Percentile(latencies, 0.5):F2}ms");
        Console.WriteLine($"P90 latency: {Percentile(latencies, 0.9):F2}ms");
        Console.WriteLine($"P99 latency: {Percentile(latencies, 0.99):F2}ms");
    }
    
    private static double Percentile(List<double> values, double percentile)
    {
        var sorted = values.OrderBy(x => x).ToList();
        var index = (int)Math.Ceiling(percentile * sorted.Count) - 1;
        return sorted[index];
    }
}
```

### 4.9 配置扩展

```csharp
// dotnet/CudaRS.Examples/Config.cs

static class Config
{
    // ========== OpenVINO测试配置 ==========
    public const bool RunOpenVinoTests = true;
    public const bool RunOpenVinoBenchmarks = true;
    
    // 测试模型路径
    public const string TestYoloModel = @"E:\models\yolo11n.onnx";
    public const string TestYoloInt8Model = @"E:\models\yolo11n_int8.xml";
    
    // 测试图片
    public const string TestImage1 = @"E:\images\test1.jpg";
    public const string TestImage2 = @"E:\images\test2.jpg";
    public const string TestImage3 = @"E:\images\test3.jpg";
    public const string TestImage4 = @"E:\images\test4.jpg";
    
    // 测试参数
    public const int WarmupIterations = 5;
    public const int BenchmarkIterations = 100;
    public const bool EnableDetailedLogging = false;
}
```

### 4.10 主程序集成

```csharp
// dotnet/CudaRS.Examples/Program.cs

using CudaRS.Examples.Tests;
using CudaRS.Examples.Benchmarks;

if (Config.RunOpenVinoTests)
{
    Console.WriteLine("\n" + new string('=', 60));
    Console.WriteLine("OpenVINO Feature Tests");
    Console.WriteLine(new string('=', 60));
    
    // 基础功能测试
    OpenVinoBasicTests.TestBasicInference();
    OpenVinoBasicTests.TestMultipleDevices();
    await OpenVinoBasicTests.TestAsyncInference();
    
    // 动态形状测试
    OpenVinoDynamicShapeTests.TestDynamicShapes();
    OpenVinoDynamicShapeTests.TestDynamicBatchSize();
    
    // 批处理测试
    OpenVinoBatchTests.TestBatchInference();
    OpenVinoBatchTests.CompareBatchVsSingle();
    
    // 性能分析测试
    OpenVinoProfilingTests.TestProfiling();
    OpenVinoProfilingTests.TestLayerProfiling();
    
    // INT8测试
    OpenVinoInt8Tests.TestInt8Inference();
    
    // 预处理测试
    OpenVinoPreprocessTests.TestGpuPreprocessing();
    
    Console.WriteLine("\n✓ All tests completed");
}

if (Config.RunOpenVinoBenchmarks)
{
    Console.WriteLine("\n" + new string('=', 60));
    Console.WriteLine("OpenVINO Benchmarks");
    Console.WriteLine(new string('=', 60));
    
    ThroughputBenchmark.Run();
    LatencyBenchmark.Run();
}
```

---

## 五、实施计划

### 5.1 Phase 1: 基础API扩展 (1周)

**Rust端**:
1. 添加动态形状FFI绑定
2. 添加模型元信息查询
3. 添加批处理推理支持

**C#端**:
1. 实现`PartialShape`和相关类型
2. 扩展`OpenVinoModel`添加Reshape
3. 扩展`OpenVinoPipeline`添加RunBatch
4. Native层P/Invoke绑定

**测试**:
- `OpenVinoBasicTests`
- `OpenVinoDynamicShapeTests`

### 5.2 Phase 2: 高级功能 (1周)

**Rust端**:
1. 集成PrePostProcessor API
2. 添加性能分析支持
3. INT8模型加载优化

**C#端**:
1. 实现`PreprocessBuilder`
2. 实现`InferenceProfiler`
3. 添加精度配置支持

**测试**:
- `OpenVinoBatchTests`
- `OpenVinoProfilingTests`
- `OpenVinoPreprocessTests`

### 5.3 Phase 3: 测试与文档 (3-5天)

1. 完善所有测试用例
2. 性能基准测试
3. 编写API文档
4. 编写集成指南

---

## 六、测试验收标准

### 6.1 功能测试

| 测试项 | 验收标准 |
|--------|---------|
| 基础推理 | ✓ CPU/GPU正常推理 |
| 动态形状 | ✓ 支持320-1280任意尺寸 |
| 批处理 | ✓ Batch 1-8正常工作 |
| 异步推理 | ✓ 4+并发请求无错误 |
| 多设备 | ✓ CPU/GPU/AUTO正常切换 |

### 6.2 性能测试

| 指标 | 目标 |
|------|------|
| 批处理加速比 | > 2.5x (batch=8 vs single) |
| INT8加速比 | > 1.8x (INT8 vs FP32) |
| GPU预处理加速 | > 1.5x (GPU vs CPU preprocess) |
| 吞吐量(CPU) | > 50 imgs/sec (640x640, YOLO-n) |
| 吞吐量(GPU) | > 200 imgs/sec (640x640, YOLO-n) |

### 6.3 稳定性测试

- 1000次推理无内存泄漏
- 长时间运行(1小时+)无崩溃
- 异常处理覆盖所有错误路径

---

## 七、风险与挑战

### 7.1 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| C API限制 | 高 | 必要时添加C++包装层 |
| 内存管理复杂 | 中 | 详细代码审查+测试 |
| 跨平台兼容 | 中 | 条件编译+版本检测 |

### 7.2 测试资源

- 需要多种YOLO模型(v5/v8/v11)
- 需要INT8量化模型
- 需要多GPU测试环境

---

## 八、总结

本方案提供了完整的C# SDK OpenVINO集成路线图,包括:

1. **新API设计**: 动态形状、批处理、预处理、性能分析
2. **完整测试方案**: 11个测试场景+2个基准测试
3. **清晰实施计划**: 3个Phase,预计2-3周完成
4. **明确验收标准**: 功能、性能、稳定性指标

**关键收益**:
- 灵活性提升: 动态形状支持
- 性能提升: 批处理+INT8+GPU预处理
- 易用性提升: Fluent API + 详细文档
- 可维护性: 完整测试覆盖

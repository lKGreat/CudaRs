using System.Diagnostics;
using System.IO;
using System.Linq;
using CudaRS;
using CudaRS.Ocr;
using CudaRS.Yolo;
using CudaRS.Examples;
using CudaRS.Examples.Tests;

Console.WriteLine("=== CudaRS Fluent API Demo ===");

PathHelpers.EnsureCudaBinsOnPath();

// Run OpenVINO tests if enabled
if (Config.RunOpenVinoTests)
{
    Console.WriteLine();
    Console.WriteLine("=== OpenVINO Feature Tests ===");
    Case1InputInfoTest.Run();
    Case2OutputInfoTest.Run();
    Case3FixedReshapeTest.Run();
    Case4DynamicDimTest.Run();
    Case5BasicBatchTest.Run();
    Case6YoloBatchTest.Run();
    Case8PreprocessBuilderTest.Run();
    Case9GpuPreprocessTest.Run();
    Case10BasicProfilingTest.Run();
    Case11LayerProfilingTest.Run();
    Case12Int8Test.Run();
    Case13IntegrationTest.Run();
    CasePaddleOpenVinoTest.Run();
    
    Console.WriteLine("\n" + new string('=', 60));
    Console.WriteLine("All 12 OpenVINO test cases completed!");
    Console.WriteLine(new string('=', 60));
    return;
}

// 根据配置运行不同的基准测试模式
if (Config.RunAnnotationDemo)
{
    Console.WriteLine();
    Console.WriteLine("=== Image Annotation Demo ===");
    RunAnnotationDemo();
}

if (Config.OnlyFluentBench)
{
    Console.WriteLine();
    Console.WriteLine("=== Fluent OCR + YOLO Benchmark ===");
    RunFluentOcrBench();
    RunFluentYoloBench();
    return;
}

if (Config.OnlyOcr)
{
    Console.WriteLine();
    Console.WriteLine("=== PaddleOCR Test ===");
    RunOcrTest();
    return;
}

Console.WriteLine("No benchmark mode enabled. Set Config.OnlyFluentBench = true");

// ========== OCR 基准测试 ==========

static void RunOcrTest()
{
    if (!ValidateOcrConfig())
        return;

    var pipeline = CudaRsFluent.Create()
        .Pipeline()
        .ForOcr(cfg =>
        {
            cfg.DetModelDir = Config.OcrDetModelDir;
            cfg.RecModelDir = Config.OcrRecModelDir;
            cfg.Device = "cpu";
            cfg.Precision = "fp32";
            cfg.EnableMkldnn = true;
            cfg.CpuThreads = Config.CpuThreads;
            cfg.OcrVersion = Config.OcrVersion;
            cfg.TextDetectionModelName = Config.OcrDetModelName;
            cfg.TextRecognitionModelName = Config.OcrRecModelName;
        })
        .AsPaddle()
        .BuildOcr();

    var bytes = File.ReadAllBytes(Config.OcrImagePath);
    var sw = Stopwatch.StartNew();
    var result = pipeline.Run(bytes);
    sw.Stop();

    Console.WriteLine($"OCR image: {Config.OcrImagePath}");
    Console.WriteLine($"OCR time: {sw.Elapsed.TotalMilliseconds:F2} ms");
    Console.WriteLine($"OCR lines: {result.Lines.Count}");

    var preview = string.Join(" | ", result.Lines.Select(l => l.Text).Where(t => !string.IsNullOrWhiteSpace(t)).Take(5));
    if (!string.IsNullOrWhiteSpace(preview))
        Console.WriteLine($"OCR text preview: {preview}");

    if (!string.IsNullOrWhiteSpace(result.StructJson))
        Console.WriteLine($"OCR struct json bytes: {result.StructJson.Length}");

    if (pipeline is IDisposable disposable)
        disposable.Dispose();
}

static void RunFluentOcrBench()
{
    if (!ValidateOcrConfig())
        return;

    var pipeline = CudaRsFluent.Create()
        .Pipeline()
        .ForOcr(cfg =>
        {
            cfg.DetModelDir = Config.OcrDetModelDir;
            cfg.RecModelDir = Config.OcrRecModelDir;
            cfg.Device = "cpu";
            cfg.Precision = "fp32";
            cfg.EnableMkldnn = true;
            cfg.CpuThreads = Config.CpuThreads;
            cfg.OcrVersion = Config.OcrVersion;
            cfg.TextDetectionModelName = Config.OcrDetModelName;
            cfg.TextRecognitionModelName = Config.OcrRecModelName;
        })
        .AsPaddle()
        .BuildOcr();

    var images = new List<ImageInput>
    {
        new(Config.OcrImagePath, File.ReadAllBytes(Config.OcrImagePath))
    };

    var result = BenchmarkHelpers.RunBenchmark(
        "Fluent-OCR-Paddle",
        pipeline.Run,
        images,
        Config.PipelineIterations,
        Config.OpenVinoWarmupIterations,
        (ocrResult, iter) =>
        {
            if (iter == 0 && Config.DetailedOutput)
            {
                var preview = string.Join(" | ", ocrResult.Lines.Select(l => l.Text).Where(t => !string.IsNullOrWhiteSpace(t)).Take(5));
                if (!string.IsNullOrWhiteSpace(preview))
                    Console.WriteLine($"  OCR preview: {preview}");
            }
        });

    BenchmarkHelpers.PrintStats(result);

    if (pipeline is IDisposable disposable)
        disposable.Dispose();
}

// ========== YOLO 基准测试 ==========

static void RunFluentYoloBench()
{
    var onnxPaths = PathHelpers.FindModels(Config.OnnxModelPaths, ".onnx");
    if (onnxPaths.Count == 0)
    {
        Console.WriteLine("Fluent YOLO bench skipped: no ONNX models found.");
        return;
    }

    var imageInputs = PathHelpers.LoadImages(Config.ImagePaths);
    if (imageInputs.Count == 0)
    {
        Console.WriteLine("Fluent YOLO bench skipped: no input images found.");
        return;
    }

    Console.WriteLine();
    Console.WriteLine("=== Fluent YOLO Benchmark (OpenVINO) ===");
    Console.WriteLine($"Models: {onnxPaths.Count}, Images: {imageInputs.Count}");

    foreach (var onnxPath in onnxPaths)
    {
        var labels = PathHelpers.LoadLabels(onnxPath, Config.LabelsPath);
        var version = Config.OnnxVersions.Length > 0 ? Config.OnnxVersions[0] : YoloVersion.V8;
        var task = Config.OnnxTasks.Length > 0 ? Config.OnnxTasks[0] : YoloTask.Detect;

        foreach (var device in Config.OpenVinoYoloDevices)
        {
            Console.WriteLine();
            Console.WriteLine($"[YOLO] model={Path.GetFileName(onnxPath)} device={device}");

            try
            {
                var builder = CudaRsFluent.Create()
                    .Pipeline()
                    .ForYolo(onnxPath, cfg =>
                    {
                        cfg.Version = version;
                        cfg.Task = task;
                        cfg.InputWidth = Config.InputWidth;
                        cfg.InputHeight = Config.InputHeight;
                        cfg.InputChannels = Config.InputChannels;
                        cfg.ConfidenceThreshold = Config.ConfidenceThreshold;
                        cfg.IouThreshold = Config.IouThreshold;
                        cfg.MaxDetections = Config.MaxDetections;
                        cfg.ClassNames = labels;
                        YoloVersionAdapter.ApplyVersionDefaults(cfg);
                    })
                    .WithThroughput(opts =>
                    {
                        opts.BatchSize = 1;
                        opts.NumStreams = 1;
                        opts.MaxBatchDelayMs = 2;
                    });

                builder = device.Equals("cpu", StringComparison.OrdinalIgnoreCase)
                    ? builder.AsCpu()
                    : builder.AsOpenVino();

                var pipeline = builder.BuildYolo();

                var result = BenchmarkHelpers.RunBenchmark(
                    $"YOLO-{device}",
                    pipeline.Run,
                    imageInputs,
                    Config.PipelineIterations,
                    Config.OpenVinoWarmupIterations,
                    (inferResult, iter) =>
                    {
                        if (Config.DetailedOutput && iter == 0)
                        {
                            var detIndex = 0;
                            foreach (var det in inferResult.Detections)
                            {
                                detIndex++;
                                if (detIndex > Config.MaxDetectionsToPrint)
                                    break;
                                Console.WriteLine($"  Det {detIndex}: {det}");
                            }
                        }
                    });

                BenchmarkHelpers.PrintStats(result);

                if (pipeline is IDisposable disposable)
                    disposable.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[YOLO] device={device} failed: {ex.Message}");
            }
        }
    }
}

// ========== 验证辅助方法 ==========

static void RunAnnotationDemo()
{
    var onnxPaths = PathHelpers.FindModels(Config.OnnxModelPaths, ".onnx");
    if (onnxPaths.Count == 0)
    {
        Console.WriteLine("Annotation demo skipped: no ONNX models found.");
        return;
    }

    var imageInputs = PathHelpers.LoadImages(Config.ImagePaths);
    if (imageInputs.Count == 0)
    {
        Console.WriteLine("Annotation demo skipped: no input images found.");
        return;
    }

    var onnxPath = onnxPaths[0];
    var imageInput = imageInputs[0];
    var labels = PathHelpers.LoadLabels(onnxPath, Config.LabelsPath);

    Console.WriteLine($"Model: {Path.GetFileName(onnxPath)}");
    Console.WriteLine($"Image: {Path.GetFileName(imageInput.Path)}");

    try
    {
        // 示例1: 纯检测数据
        Console.WriteLine("\n--- Example 1: Pure Detection Data ---");
        var pipeline1 = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(onnxPath, cfg =>
            {
                cfg.Version = YoloVersion.V8;
                cfg.Task = YoloTask.Detect;
                cfg.InputWidth = Config.InputWidth;
                cfg.InputHeight = Config.InputHeight;
                cfg.ConfidenceThreshold = Config.ConfidenceThreshold;
                cfg.IouThreshold = Config.IouThreshold;
                cfg.ClassNames = labels;
            })
            .AsOpenVino()
            .BuildYoloFluent();

        var result1 = pipeline1.Run(imageInput.Bytes).AsDetections();
        Console.WriteLine($"Detected {result1.Detections.Count} objects");
        foreach (var det in result1.Detections.Take(3))
            Console.WriteLine($"  - {det}");

        // 示例2: 带框图片
        Console.WriteLine("\n--- Example 2: Annotated Image ---");
        var result2 = pipeline1.Run(imageInput.Bytes).AsAnnotatedImage(
            new AnnotationOptions
            {
                ShowLabel = true,
                ShowConfidence = true,
                BoxThickness = 3f
            },
            ImageFormat.Jpeg);

        Console.WriteLine($"Annotated image: {result2.Width}x{result2.Height}, {result2.ImageBytes.Length} bytes");

        if (Config.SaveAnnotatedImages && !string.IsNullOrEmpty(Config.AnnotatedOutputDir))
        {
            Directory.CreateDirectory(Config.AnnotatedOutputDir);
            var outputPath = Path.Combine(Config.AnnotatedOutputDir, $"annotated_{Path.GetFileName(imageInput.Path)}");
            File.WriteAllBytes(outputPath, result2.ImageBytes);
            Console.WriteLine($"Saved to: {outputPath}");
        }

        // 示例3: 两者都要
        Console.WriteLine("\n--- Example 3: Combined Result ---");
        var result3 = pipeline1.Run(imageInput.Bytes).AsCombined(
            new AnnotationOptions { ShowLabel = true },
            ImageFormat.Png);

        Console.WriteLine($"Detections: {result3.Inference.Detections.Count}");
        Console.WriteLine($"Annotated image: {result3.AnnotatedImage?.ImageBytes.Length ?? 0} bytes");

        if (Config.SaveAnnotatedImages && result3.AnnotatedImage != null && !string.IsNullOrEmpty(Config.AnnotatedOutputDir))
        {
            var outputPath = Path.Combine(Config.AnnotatedOutputDir, $"combined_{Path.GetFileName(imageInput.Path)}");
            File.WriteAllBytes(outputPath, result3.AnnotatedImage.ImageBytes);
            Console.WriteLine($"Saved to: {outputPath}");
        }

        pipeline1.Dispose();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\n[ERROR] Annotation demo failed:");
        if (Config.ShowErrorDetails)
        {
            Console.WriteLine(ex.ToString());
        }
        else
        {
            Console.WriteLine(ex.Message);
        }
    }
}

static bool ValidateOcrConfig()
{
    if (string.IsNullOrWhiteSpace(Config.OcrDetModelDir) || !Directory.Exists(Config.OcrDetModelDir))
    {
        Console.WriteLine($"OCR det model dir not found: {Config.OcrDetModelDir}");
        return false;
    }
    if (string.IsNullOrWhiteSpace(Config.OcrRecModelDir) || !Directory.Exists(Config.OcrRecModelDir))
    {
        Console.WriteLine($"OCR rec model dir not found: {Config.OcrRecModelDir}");
        return false;
    }
    if (string.IsNullOrWhiteSpace(Config.OcrImagePath) || !File.Exists(Config.OcrImagePath))
    {
        Console.WriteLine($"OCR image not found: {Config.OcrImagePath}");
        return false;
    }
    return true;
}

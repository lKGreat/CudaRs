using System.Diagnostics;
using CudaRS;
using CudaRS.Ocr;
using CudaRS.Yolo;
using CudaRS.Examples;

Console.WriteLine("=== CudaRS Fluent API Demo ===");

PathHelpers.EnsureCudaBinsOnPath();

// 根据配置运行不同的基准测试模式
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

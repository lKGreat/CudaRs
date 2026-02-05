using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using CudaRS;
using CudaRS.OpenVino;
using CudaRS.Ocr;
using CudaRS.Paddle;
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

if (Config.OnlyOpenVinoOcr)
{
    Console.WriteLine();
    Console.WriteLine("=== OpenVINO OCR (CPU) ===");
    RunOpenVinoOcrOnly();
    return;
}

if (Config.RunBackendSmoke)
{
    Console.WriteLine();
    Console.WriteLine("=== Backend Smoke Tests (OpenVINO / TensorRT / ONNXRuntime / Paddle) ===");
    RunBackendSmoke();
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

static void RunOpenVinoOcrOnly()
{
    try
    {
        Console.WriteLine("[OpenVINO OCR]...");

        if (string.IsNullOrWhiteSpace(Config.OpenVinoOcrDictPath) || !File.Exists(Config.OpenVinoOcrDictPath))
        {
            Console.WriteLine($"  FAIL: OCR dict not found: {Config.OpenVinoOcrDictPath}");
            return;
        }

        string detXml;
        string recXml;

        if (Config.OpenVinoOcrAutoConvert)
        {
            var converter = new PaddleToIrConverter(
                pythonPath: null,
                onnxCacheDir: null,
                irCacheDir: string.IsNullOrWhiteSpace(Config.OpenVinoIrCacheDir) ? null : Config.OpenVinoIrCacheDir);

            if (!converter.IsReady())
            {
                Console.WriteLine("  FAIL: Python OpenVINO or paddle2onnx not installed.");
                Console.WriteLine(PaddleToIrConverter.GetInstallationInstructions());
                return;
            }

            (detXml, recXml) = converter.ConvertOrUseCache(
                Config.OcrDetModelDir,
                Config.OcrRecModelDir,
                opsetVersion: 11,
                forceReconvert: Config.OpenVinoOcrForceReconvert,
                compressToFp16: Config.OpenVinoOcrCompressToFp16);
        }
        else
        {
            detXml = Config.OpenVinoOcrDetModelPath;
            recXml = Config.OpenVinoOcrRecModelPath;
        }

        if (!File.Exists(detXml) || !File.Exists(recXml))
        {
            Console.WriteLine($"  FAIL: OpenVINO IR model not found: {detXml} / {recXml}");
            return;
        }

        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForOpenVinoOcr(cfg =>
            {
                cfg.DetModelPath = detXml;
                cfg.RecModelPath = recXml;
                cfg.DictPath = Config.OpenVinoOcrDictPath;
                cfg.DetResizeLong = Config.OcrDetResizeLong;
                cfg.DetStride = Config.OcrDetStride;
                cfg.DetThresh = Config.OcrDetThresh;
                cfg.DetBoxThresh = Config.OcrDetBoxThresh;
                cfg.DetMaxCandidates = Config.OcrDetMaxCandidates;
                cfg.DetMinArea = Config.OcrDetMinArea;
                cfg.DetBoxPadding = Config.OcrDetBoxPadding;
                cfg.RecInputH = Config.OcrRecInputH;
                cfg.RecInputW = Config.OcrRecInputW;
                cfg.RecBatchSize = Config.OpenVinoOcrRecBatchSize;
            }, pipe =>
            {
                pipe.OpenVinoDevice = Config.OpenVinoDevice;
                pipe.OpenVinoConfigJson = Config.OpenVinoConfigJson;
                pipe.OpenVinoPerformanceMode = "throughput";
                pipe.OpenVinoEnableMmap = true;
                pipe.OpenVinoCacheDir = string.IsNullOrWhiteSpace(Config.OpenVinoIrCacheDir) ? "" : Config.OpenVinoIrCacheDir;
            })
            .AsCpu()
            .BuildOcr();

        var ocrImages = PathHelpers.LoadImages(Config.OpenVinoOcrImagePaths);
        if (ocrImages.Count == 0 && File.Exists(Config.OcrImagePath))
            ocrImages.Add(new ImageInput(Config.OcrImagePath, File.ReadAllBytes(Config.OcrImagePath)));

        if (ocrImages.Count == 0)
        {
            Console.WriteLine($"  FAIL: OCR images not found. OpenVinoOcrImagePaths empty and OcrImagePath missing: {Config.OcrImagePath}");
            return;
        }

        for (var i = 0; i < Config.OpenVinoOcrWarmupIterations; i++)
        {
            _ = pipeline.Run(ocrImages[0].Bytes);
        }

        Console.WriteLine($"  Images: {ocrImages.Count}");
        foreach (var image in ocrImages)
        {
            var (ocr, ms) = Timed(() => pipeline.Run(image.Bytes));
            Console.WriteLine($"  {Path.GetFileName(image.Path)}: {ocr.Lines.Count} lines, time={ms:F2} ms");
        }

        if (pipeline is IDisposable disposable)
            disposable.Dispose();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  FAIL: {ex.Message}");
    }
}

static void RunBackendSmoke()
{
    var onnxModel = File.Exists(Config.TestYoloModel) ? Config.TestYoloModel : null;
    var trtEngine = File.Exists(Config.LegacyEnginePath) ? Config.LegacyEnginePath : null;
    var imagePath = File.Exists(Config.TestImage1) ? Config.TestImage1 : null;

    if (onnxModel == null)
    {
        Console.WriteLine($"[YOLO] ONNX model not found: {Config.TestYoloModel}");
        return;
    }

    if (imagePath == null)
    {
        Console.WriteLine($"[YOLO] Image not found: {Config.TestImage1}");
        return;
    }

    var imageBytes = File.ReadAllBytes(imagePath);
    var labels = PathHelpers.LoadLabels(onnxModel, Config.LabelsPath);

    // OpenVINO driver check
    RunOpenVinoDriverCheck();

    // OpenVINO YOLO
    try
    {
        Console.WriteLine("[OpenVINO] YOLO...");
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(onnxModel, cfg =>
            {
                cfg.Version = Config.OnnxVersions.Length > 0 ? Config.OnnxVersions[0] : YoloVersion.V8;
                cfg.Task = Config.OnnxTasks.Length > 0 ? Config.OnnxTasks[0] : YoloTask.Detect;
                cfg.InputWidth = Config.InputWidth;
                cfg.InputHeight = Config.InputHeight;
                cfg.InputChannels = Config.InputChannels;
                cfg.ConfidenceThreshold = Config.ConfidenceThreshold;
                cfg.IouThreshold = Config.IouThreshold;
                cfg.MaxDetections = Config.MaxDetections;
                cfg.ClassNames = labels;
                YoloVersionAdapter.ApplyVersionDefaults(cfg);
            })
            .AsCpu()
            .BuildYolo();

        var (result, ms) = Timed(() => pipeline.Run(imageBytes));
        Console.WriteLine($"  OK: {result.Detections.Count} detections, time={ms:F2} ms");
        if (pipeline is IDisposable disposable)
            disposable.Dispose();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  FAIL: {ex.Message}");
    }

    // ONNX Runtime YOLO (CPU pipeline)
    try
    {
        Console.WriteLine("[ONNXRuntime] YOLO...");
        var pipeline = CudaRsFluent.Create()
            .Pipeline()
            .ForYolo(onnxModel, cfg =>
            {
                cfg.Version = Config.OnnxVersions.Length > 0 ? Config.OnnxVersions[0] : YoloVersion.V8;
                cfg.Task = Config.OnnxTasks.Length > 0 ? Config.OnnxTasks[0] : YoloTask.Detect;
                cfg.InputWidth = Config.InputWidth;
                cfg.InputHeight = Config.InputHeight;
                cfg.InputChannels = Config.InputChannels;
                cfg.ConfidenceThreshold = Config.ConfidenceThreshold;
                cfg.IouThreshold = Config.IouThreshold;
                cfg.MaxDetections = Config.MaxDetections;
                cfg.ClassNames = labels;
                YoloVersionAdapter.ApplyVersionDefaults(cfg);
            })
            .AsCpu()
            .BuildYolo();

        var (result, ms) = Timed(() => pipeline.Run(imageBytes));
        Console.WriteLine($"  OK: {result.Detections.Count} detections, time={ms:F2} ms");
        if (pipeline is IDisposable disposable)
            disposable.Dispose();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  FAIL: {ex.Message}");
    }

    // TensorRT YOLO
    if (trtEngine == null)
    {
        Console.WriteLine($"[TensorRT] engine not found: {Config.LegacyEnginePath}");
    }
    else
    {
        try
        {
            Console.WriteLine("[TensorRT] YOLO...");
            var pipeline = CudaRsFluent.Create()
                .Pipeline()
                .ForYolo(trtEngine, cfg =>
                {
                    cfg.Version = Config.Versions.Length > 0 ? Config.Versions[0] : YoloVersion.V8;
                    cfg.Task = Config.Tasks.Length > 0 ? Config.Tasks[0] : YoloTask.Detect;
                    cfg.InputWidth = Config.InputWidth;
                    cfg.InputHeight = Config.InputHeight;
                    cfg.InputChannels = Config.InputChannels;
                    cfg.ConfidenceThreshold = Config.ConfidenceThreshold;
                    cfg.IouThreshold = Config.IouThreshold;
                    cfg.MaxDetections = Config.MaxDetections;
                    cfg.ClassNames = labels;
                    YoloVersionAdapter.ApplyVersionDefaults(cfg);
                })
                .AsTensorRt()
                .BuildYolo();

            var (result, ms) = Timed(() => pipeline.Run(imageBytes));
            Console.WriteLine($"  OK: {result.Detections.Count} detections, time={ms:F2} ms");
            if (pipeline is IDisposable disposable)
                disposable.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  FAIL: {ex.Message}");
        }
    }

    // Paddle OCR
    if (Config.RunPaddleOcrSmoke)
    {
        if (!ValidateOcrConfig())
            return;

        try
        {
            Console.WriteLine("[PaddleOCR]...");
            var pipeline = CudaRsFluent.Create()
                .Pipeline()
                .ForOcr(cfg =>
                {
                    cfg.DetModelDir = Config.OcrDetModelDir;
                    cfg.RecModelDir = Config.OcrRecModelDir;
                    cfg.Device = "cpu";
                    cfg.Precision = "fp32";
                    cfg.EnableMkldnn = true;
                    cfg.CpuThreads = Math.Max(1, Environment.ProcessorCount);
                    cfg.MkldnnCacheCapacity = 20;
                    cfg.OcrVersion = Config.OcrVersion;
                    cfg.TextDetectionModelName = Config.OcrDetModelName;
                    cfg.TextRecognitionModelName = Config.OcrRecModelName;
                    cfg.TextDetLimitSideLen = 960;
                    cfg.TextDetLimitType = "max";
                    cfg.TextRecognitionBatchSize = 8;
                })
                .AsPaddle()
                .BuildOcr();

            var ocrInput = File.ReadAllBytes(Config.OcrImagePath);
            var iterations = 10;
            for (var i = 1; i <= iterations; i++)
            {
                var (ocr, ms) = Timed(() => pipeline.Run(ocrInput));
                Console.WriteLine($"  Iter {i}/{iterations}: {ocr.Lines.Count} lines, time={ms:F2} ms");
            }
            if (pipeline is IDisposable disposable)
                disposable.Dispose();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  FAIL: {ex.Message}");
        }
    }

    // OpenVINO OCR
    if (Config.RunOpenVinoOcr)
        RunOpenVinoOcrOnly();
}

static void RunOpenVinoDriverCheck()
{
    Console.WriteLine("[OpenVINO] Driver check...");
    var baseDir = AppContext.BaseDirectory;

    CheckDll("openvino.dll", baseDir);
    CheckDll("openvino_c.dll", baseDir);
    CheckDll("openvino_intel_cpu_plugin.dll", baseDir);
    CheckDll("openvino_intel_gpu_plugin.dll", baseDir);
}

static void CheckDll(string fileName, string baseDir)
{
    var localPath = Path.Combine(baseDir, fileName);
    var pathToLoad = File.Exists(localPath) ? localPath : fileName;
    var ok = TryLoadLibrary(pathToLoad, out var lastError);
    var tag = ok ? "OK" : $"FAIL (err={lastError})";
    Console.WriteLine($"  {fileName}: {tag}");
}

static bool TryLoadLibrary(string path, out int lastError)
{
    var handle = LoadLibrary(path);
    if (handle == IntPtr.Zero)
    {
        lastError = Marshal.GetLastWin32Error();
        return false;
    }

    FreeLibrary(handle);
    lastError = 0;
    return true;
}

[DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
static extern IntPtr LoadLibrary(string lpFileName);

[DllImport("kernel32", SetLastError = true)]
static extern bool FreeLibrary(IntPtr hModule);

static (T Result, double Ms) Timed<T>(Func<T> action)
{
    var sw = Stopwatch.StartNew();
    var result = action();
    sw.Stop();
    return (result, sw.Elapsed.TotalMilliseconds);
}

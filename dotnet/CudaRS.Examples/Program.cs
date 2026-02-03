using CudaRS;
using CudaRS.Core;
using CudaRS.Yolo;
using System.Runtime.InteropServices;

Console.WriteLine("=== CudaRS Example ===\n");

// Get library version
Console.WriteLine($"CudaRS Version: {Cuda.Version}");

static bool HasExport(string name)
{
    if (!NativeLibrary.TryLoad("cudars_ffi", out var handle))
        return false;
    try
    {
        return NativeLibrary.TryGetExport(handle, name, out _);
    }
    finally
    {
        NativeLibrary.Free(handle);
    }
}

// Check CUDA devices
try
{
    var deviceCount = Cuda.DeviceCount;
    Console.WriteLine($"CUDA Devices Found: {deviceCount}");

    if (deviceCount > 0)
    {
        // Get driver version
        var driverVersion = CudaDriver.Version;
        Console.WriteLine($"CUDA Driver Version: {driverVersion / 1000}.{(driverVersion % 1000) / 10}");

        // Set device 0
        Cuda.CurrentDevice = 0;
        Console.WriteLine($"Using Device: {Cuda.CurrentDevice}");

        // Test memory allocation
        Console.WriteLine("\n--- Memory Test ---");
        const int arraySize = 1024;
        var hostData = new float[arraySize];
        for (int i = 0; i < arraySize; i++)
            hostData[i] = i * 0.5f;

        using var deviceBuffer = hostData.ToDevice();
        Console.WriteLine($"Allocated {arraySize} floats on device");

        // Copy back and verify
        var resultData = deviceBuffer.ToArray();
        var correct = true;
        for (int i = 0; i < arraySize && correct; i++)
            correct = Math.Abs(hostData[i] - resultData[i]) < 0.0001f;
        Console.WriteLine($"Data verification: {(correct ? "PASSED" : "FAILED")}");

        // Test streams and events
        Console.WriteLine("\n--- Stream & Event Test ---");
        using var stream = new CudaStream();
        using var startEvent = new CudaEvent();
        using var endEvent = new CudaEvent();

        startEvent.Record(stream);
        // Simulate some work
        Thread.Sleep(10);
        endEvent.Record(stream);
        stream.Synchronize();

        var elapsedMs = endEvent.ElapsedTime(startEvent);
        Console.WriteLine($"Elapsed time: {elapsedMs:F3} ms");

        // Test library handles
        Console.WriteLine("\n--- Library Handles Test ---");
        if (HasExport("cudars_cublas_create"))
        {
            try
            {
                using var cublasHandle = new CublasHandle();
                Console.WriteLine($"cuBLAS Version: {cublasHandle.Version}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"cuBLAS: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("cuBLAS: not available");
        }

        if (HasExport("cudars_cudnn_get_version"))
        {
            try
            {
                Console.WriteLine($"cuDNN Version: {CudnnHandle.Version}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"cuDNN: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("cuDNN: not available");
        }

        if (HasExport("cudars_nvrtc_version"))
        {
            try
            {
                var (major, minor) = Nvrtc.Version;
                Console.WriteLine($"NVRTC Version: {major}.{minor}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"NVRTC: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("NVRTC: not available");
        }

        // Synchronize device
        Cuda.Synchronize();
    }
}
catch (CudaException ex)
{
    Console.WriteLine($"CUDA Error: {ex.ErrorCode} - {ex.Message}");
}

// GPU Management via NVML
Console.WriteLine("\n--- GPU Management (NVML) ---");
if (HasExport("cudars_nvml_init"))
{
    try
    {
        var gpuCount = GpuManagement.DeviceCount;
        Console.WriteLine($"GPUs Found: {gpuCount}");

        for (uint i = 0; i < gpuCount; i++)
        {
            Console.WriteLine($"\nGPU {i}:");
            
            var memInfo = GpuManagement.GetMemoryInfo(i);
            Console.WriteLine($"  Memory: {memInfo.Used / (1024 * 1024)} MB / {memInfo.Total / (1024 * 1024)} MB");

            var utilRates = GpuManagement.GetUtilizationRates(i);
            Console.WriteLine($"  GPU Utilization: {utilRates.Gpu}%");
            Console.WriteLine($"  Memory Utilization: {utilRates.Memory}%");

            var temp = GpuManagement.GetTemperature(i);
            Console.WriteLine($"  Temperature: {temp}°C");

            try
            {
                var power = GpuManagement.GetPowerUsage(i);
                Console.WriteLine($"  Power: {power / 1000.0:F1} W");
            }
            catch { /* Power reading not available */ }

            try
            {
                var fan = GpuManagement.GetFanSpeed(i);
                Console.WriteLine($"  Fan Speed: {fan}%");
            }
            catch { /* Fan speed not available */ }
        }

        GpuManagement.Shutdown();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"NVML Error: {ex.Message}");
    }
}
else
{
    Console.WriteLine("NVML: not available");
}

Console.WriteLine("\n=== Example Complete ===");

// Fluent demo layer
Console.WriteLine("\n=== Fluent Pipeline Demo ===\n");

try
{
    if (!HasExport("cudars_memory_pool_create_with_device"))
    {
        Console.WriteLine("Fluent demo skipped: native export 'cudars_memory_pool_create_with_device' not found.");
    }
    else
    {
        var pipeline = InferencePipeline.Create()
            .WithName("SecurityMultiChannel")
            .WithChannel("cam-1", channel => channel
                .WithShape(1920, 1080, 3)
                .WithDataType("uint8")
                .WithScenePriority(SceneLevel.L2)
                .WithFpsRange(5, 25)
                .WithBatching(1, 0))
            .WithChannel("cam-2", channel => channel
                .WithShape(1280, 720, 3)
                .WithDataType("uint8")
                .WithScenePriority(SceneLevel.L1)
                .WithFpsRange(5, 20)
                .WithBatching(1, 0))
            .WithModel("detector", model => model
                .FromPath("models/detector.onnx")
                .WithBackend("onnx")
                .OnDevice("cuda:0")
                .WithPrecision("fp16"))
            .WithPreprocessStage("resize+normalize")
            .WithInferStage("detector")
            .WithPostprocessStage("nms")
            .WithExecution(options => options
                .WithMaxConcurrency(2)
                .WithStreamMode(StreamMode.Async)
                .WithStreamPoolSize(32)
                .WithMaxQueueDepth(5000))
            .Build();

        var demoInputs = new Dictionary<string, ChannelInput>
        {
            ["cam-1"] = new ChannelInput(new byte[0]) { SceneLevel = SceneLevel.L2 },
            ["cam-2"] = new ChannelInput(new byte[0]) { SceneLevel = SceneLevel.L1 },
        };

        var demoResult = pipeline.Run(new PipelineInput(demoInputs));
        Console.WriteLine($"Pipeline: {demoResult.PipelineName}");
        Console.WriteLine($"Success: {demoResult.Success}");
        Console.WriteLine($"Elapsed: {demoResult.Elapsed.TotalMilliseconds:F3} ms");
        if (demoResult.Diagnostics.Count > 0)
        {
            Console.WriteLine("Diagnostics:");
            foreach (var message in demoResult.Diagnostics)
                Console.WriteLine($"- {message}");
        }
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Fluent demo failed (ignored): {ex.Message}");
}

// YOLO inference example (C# layer)
Console.WriteLine("\n=== YOLO Inference Example ===\n");

var enginePath = @"E:\codeding\AI\onnx\best\best.engine";
var modelDir = Path.GetDirectoryName(enginePath) ?? string.Empty;
var yoloModelPath = Path.Combine(modelDir, "best.onnx");
var trtExecPath = Environment.GetEnvironmentVariable("TRTEXEC_PATH")
    ?? @"E:\codeding\AI\TensorRT-10.15.1.29.Windows.amd64.cuda-12.9\TensorRT-10.15.1.29\bin\trtexec.exe";

var labelsPath = Path.Combine(modelDir, "labels.txt");
    if (!File.Exists(enginePath))
    {
        if (!File.Exists(yoloModelPath))
        {
            Console.WriteLine($"Engine not found: {enginePath}");
            Console.WriteLine($"ONNX model not found: {yoloModelPath}");
            return;
        }
        if (!File.Exists(trtExecPath))
        {
            Console.WriteLine($"trtexec not found: {trtExecPath}");
            return;
        }

        Console.WriteLine("Building TensorRT engine with trtexec...");
        var exitCode = TensorRtExec.BuildEngine(
            trtExecPath,
            yoloModelPath,
            enginePath,
            workspaceMb: 1024,
            fp16: true);
        if (exitCode != 0 || !File.Exists(enginePath))
        {
            Console.WriteLine($"trtexec failed, exit code: {exitCode}");
            return;
        }
    }

if (!File.Exists(labelsPath))
{
    var alt = Path.Combine(modelDir, "classes.txt");
    if (File.Exists(alt))
        labelsPath = alt;
}

string? imagePath = null;
if (Directory.Exists(modelDir))
{
    var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png", ".bmp" };
    imagePath = Directory.EnumerateFiles(modelDir)
        .FirstOrDefault(p => exts.Contains(Path.GetExtension(p)));
}

if (!File.Exists(enginePath))
{
    Console.WriteLine($"Engine not found: {enginePath}.");
}
else if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
{
    Console.WriteLine($"Image not found in: {modelDir}. Place a test image in the same folder.");
}
else
{
    var classNames = File.Exists(labelsPath) ? YoloLabels.LoadFromFile(labelsPath) : Array.Empty<string>();

    var hasTensorRt = HasExport("cudars_trt_build_engine");
    if (!hasTensorRt)
        Console.WriteLine("TensorRT 未启用，示例将回退到 ONNX Runtime（CPU）。如需 CUDA 推理请安装 TensorRT 并重新编译 cudars_ffi --features tensorrt。");

    var yoloConfig = new YoloConfig
    {
        Version = YoloVersion.V8,
        Task = YoloTask.Detect,
        Backend = hasTensorRt ? InferenceBackend.TensorRT : InferenceBackend.OnnxRuntime,
        InputWidth = 640,
        InputHeight = 640,
        InputChannels = 3,
        ConfidenceThreshold = 0.25f,
        IouThreshold = 0.45f,
        MaxDetections = 100,
        ClassNames = classNames,
    };

    var yoloDef = new YoloModelDefinition
    {
        ModelId = "yolo-best",
        ModelPath = enginePath,
        Config = yoloConfig,
        DeviceId = 0,
    };

    // 图片加载计时
    var imageLoadStopwatch = System.Diagnostics.Stopwatch.StartNew();
    var image = YoloImage.FromFile(imagePath);
    imageLoadStopwatch.Stop();
    var imageLoadTime = imageLoadStopwatch.Elapsed;

    // 模型加载 + 预热计时（包含 CUDA kernel 编译等开销）
    var loadStopwatch = System.Diagnostics.Stopwatch.StartNew();
    using var yolo = YoloModel.Create(yoloDef);
    Console.WriteLine("Warming up...");
    _ = yolo.Run("warmup", image, frameIndex: 0);
    loadStopwatch.Stop();
    var modelLoadTime = loadStopwatch.Elapsed;
    
    // 预处理一次（后续复用）
    var preprocessStopwatch = System.Diagnostics.Stopwatch.StartNew();
    var preprocess = YoloPreprocessor.Letterbox(image, yoloConfig.InputWidth, yoloConfig.InputHeight);
    preprocessStopwatch.Stop();
    var preprocessTime = preprocessStopwatch.Elapsed;
    
    // 开始计时 - 总计时
    var totalStopwatch = System.Diagnostics.Stopwatch.StartNew();
    
    // 正式推理计时（多次取平均）- 只计算纯推理时间
    const int inferenceRuns = 100;
    var inferStopwatch = System.Diagnostics.Stopwatch.StartNew();
    BackendResult backendResult = null!;
    for (int i = 0; i < inferenceRuns; i++)
    {
        backendResult = yolo.RunRaw(preprocess.Input, preprocess.InputShape);
    }
    inferStopwatch.Stop();
    var inferenceTime = inferStopwatch.Elapsed;
    var avgInferenceTime = inferenceTime.TotalMilliseconds / inferenceRuns;
    
    // 后处理计时
    var postprocessStopwatch = System.Diagnostics.Stopwatch.StartNew();
    var yoloResult = YoloPostprocessor.Decode(yoloDef.ModelId, yoloConfig, backendResult, preprocess, "demo", 1);
    postprocessStopwatch.Stop();
    var postprocessTime = postprocessStopwatch.Elapsed;
    
    // 总计时结束
    totalStopwatch.Stop();
    var totalTime = totalStopwatch.Elapsed;

    Console.WriteLine($"YOLO Success: {yoloResult.Success}");
    Console.WriteLine($"Detections: {yoloResult.Detections.Count}");
    
    // 输出时间统计
    Console.WriteLine($"\n--- 时间统计 ---");
    Console.WriteLine($"图片加载时间:       {imageLoadTime.TotalMilliseconds:F2} ms");
    Console.WriteLine($"模型加载+预热时间:  {modelLoadTime.TotalMilliseconds:F2} ms");
    Console.WriteLine($"预处理时间:         {preprocessTime.TotalMilliseconds:F2} ms");
    Console.WriteLine($"纯推理时间 (x{inferenceRuns}):  {inferenceTime.TotalMilliseconds:F2} ms");
    Console.WriteLine($"平均纯推理时间:     {avgInferenceTime:F2} ms");
    Console.WriteLine($"后处理时间:         {postprocessTime.TotalMilliseconds:F2} ms");
    Console.WriteLine($"纯推理总时间:       {totalTime.TotalMilliseconds:F2} ms");
    if (yoloResult.Nms != null)
    {
        Console.WriteLine($"NMS: pre={yoloResult.Nms.PreNmsCount} post={yoloResult.Nms.PostNmsCount} iou={yoloResult.Nms.IouThreshold:F2} max={yoloResult.Nms.MaxDetections} classAgnostic={yoloResult.Nms.ClassAgnostic}");
    }

    foreach (var det in yoloResult.Detections.Take(5))
        Console.WriteLine(det);

    // --------------------------------------------------------------------
    // New: 3 pipelines sequential benchmark on the same image
    // --------------------------------------------------------------------
    Console.WriteLine("\n=== 3-Pipeline Parallel Benchmark ===\n");
    const int pipelineCount = 3;
    const int repeatRuns = 10;
    var pipelines = new List<YoloModel>(pipelineCount);
    try
    {
        for (int i = 0; i < pipelineCount; i++)
        {
            var model = YoloModel.Create(yoloDef);
            _ = model.Run($"warmup-p{i + 1}", image, frameIndex: 0);
            pipelines.Add(model);
        }

        var perPipelineTimes = new double[pipelineCount][];
        for (int p = 0; p < pipelineCount; p++)
            perPipelineTimes[p] = new double[repeatRuns];
        var lastResults = new ModelInferenceResult?[pipelineCount];
        var totalSw = System.Diagnostics.Stopwatch.StartNew();

        for (int r = 0; r < repeatRuns; r++)
        {
            var tasks = new Task[pipelineCount];
            for (int p = 0; p < pipelineCount; p++)
            {
                var pipelineIndex = p;
                tasks[p] = Task.Run(() =>
                {
                    var start = System.Diagnostics.Stopwatch.GetTimestamp();
                    var res = pipelines[pipelineIndex].Run($"pipe-{pipelineIndex + 1}", image, frameIndex: r);
                    perPipelineTimes[pipelineIndex][r] = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                    lastResults[pipelineIndex] = res;
                });
            }

            Task.WaitAll(tasks);
        }

        totalSw.Stop();

        for (int p = 0; p < pipelineCount; p++)
        {
            var totalMs = perPipelineTimes[p].Sum();
            Console.WriteLine($"Pipeline {p + 1} (runs={repeatRuns}) total: {totalMs:F2} ms");
            for (int r = 0; r < repeatRuns; r++)
                Console.WriteLine($"  Run {r + 1}: {perPipelineTimes[p][r]:F2} ms");

            var last = lastResults[p];
            if (last != null)
                Console.WriteLine($"  Last result: success={last.Success}, detections={last.Detections.Count}");
        }

        Console.WriteLine($"All pipelines wall time: {totalSw.Elapsed.TotalMilliseconds:F2} ms");
    }
    finally
    {
        foreach (var model in pipelines)
            model.Dispose();
    }

    // --------------------------------------------------------------------
    // New: end-to-end GPU pipeline benchmark (Rust decode + async stream)
    // --------------------------------------------------------------------

    var hasGpuPipeline =
        HasExport("cudars_image_decoder_create") &&
        HasExport("cudars_preprocess_run_device_on_stream") &&
        HasExport("cudars_trt_enqueue_device") &&
        HasExport("cudars_memcpy_dtoh_async_raw") &&
        HasExport("cudars_host_alloc_pinned");

    if (!hasGpuPipeline)
    {
        Console.WriteLine("\n[GPU Pipeline] Native exports not found. Rebuild cudars_ffi with --features tensorrt,rtc,jpeg.");
    }
    else
    {
        var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { ".jpg", ".jpeg", ".png" };
        var imageFiles = Directory.EnumerateFiles(modelDir)
            .Where(p => exts.Contains(Path.GetExtension(p)))
            .Take(64)
            .ToArray();

        if (imageFiles.Length == 0)
        {
            Console.WriteLine("\n[GPU Pipeline] No .jpg/.png found to benchmark.");
        }
        else
        {
            var images = imageFiles.Select(File.ReadAllBytes).ToArray();

            var workerCount = Math.Clamp(Environment.ProcessorCount / 2, 1, 4);
            Console.WriteLine($"\n[GPU Pipeline] workers={workerCount}, images={images.Length}");

            using var pipelineGpu = new GpuYoloThroughputPipeline(
                yoloDef,
                deviceId: 0,
                maxInputWidth: 4096,
                maxInputHeight: 4096,
                workerCount: workerCount,
                channelCapacity: 256);

            // Warmup a few frames (jit kernels + load nvjpeg + TRT context init)
            for (int i = 0; i < Math.Min(8, images.Length); i++)
                _ = await pipelineGpu.EnqueueAsync(images[i], "warmup", i);

            const int frames = 200;
            var latencies = new double[frames];

            async Task<ModelInferenceResult> TimedRun(int i)
            {
                var start = System.Diagnostics.Stopwatch.GetTimestamp();
                var res = await pipelineGpu.EnqueueAsync(images[i % images.Length], "bench", i);
                latencies[i] = System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                return res;
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var tasks = new Task<ModelInferenceResult>[frames];
            for (int i = 0; i < frames; i++)
                tasks[i] = TimedRun(i);

            var results = await Task.WhenAll(tasks);
            sw.Stop();

            var ok = results.Count(r => r.Success);
            Array.Sort(latencies);
            var avg = sw.Elapsed.TotalMilliseconds / frames;
            var fps = frames / sw.Elapsed.TotalSeconds;
            var p50 = latencies[(int)(0.50 * (frames - 1))];
            var p95 = latencies[(int)(0.95 * (frames - 1))];

            Console.WriteLine("\n--- GPU Pipeline (E2E) ---");
            Console.WriteLine($"Success: {ok}/{frames}");
            Console.WriteLine($"Total:   {sw.Elapsed.TotalMilliseconds:F2} ms");
            Console.WriteLine($"Avg:     {avg:F2} ms/frame ({fps:F1} FPS)");
            Console.WriteLine($"P50:     {p50:F2} ms");
            Console.WriteLine($"P95:     {p95:F2} ms");
        }
    }
}

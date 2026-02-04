using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CudaRS;
using CudaRS.Yolo;

Console.WriteLine("=== CudaRS TensorRT Multi-Model / Multi-Pipeline Demo ===");

EnsureCudaBinsOnPath();

if (HardcodedConfig.UseLegacyMode)
{
    await RunLegacyAsync();
    return;
}

var enginePaths = ResolveEnginePaths(HardcodedConfig.EnginePaths);
var onnxPaths = ResolveOnnxPaths(HardcodedConfig.OnnxModelPaths);
if (enginePaths.Count == 0)
    Console.WriteLine("No TensorRT engines found in hardcoded paths.");
if (onnxPaths.Count == 0)
    Console.WriteLine("No ONNX models found in hardcoded paths.");
if (enginePaths.Count == 0 && onnxPaths.Count == 0)
    return;

var imageInputs = LoadImages(enginePaths.Count > 0 ? enginePaths : onnxPaths, HardcodedConfig.ImagePaths);
if (imageInputs.Count == 0)
{
    Console.WriteLine("No input images found in hardcoded paths.");
    return;
}

var deviceId = HardcodedConfig.DeviceId;
var inputWidth = HardcodedConfig.InputWidth;
var inputHeight = HardcodedConfig.InputHeight;
var inputChannels = HardcodedConfig.InputChannels;
var cpuThreads = ResolveCpuThreads(args);
var conf = HardcodedConfig.ConfidenceThreshold;
var iou = HardcodedConfig.IouThreshold;
var maxDet = HardcodedConfig.MaxDetections;

var versions = ResolveVersions(enginePaths, HardcodedConfig.Versions);
var tasks = ResolveTasks(enginePaths, HardcodedConfig.Tasks);
var onnxVersions = ResolveVersions(onnxPaths, HardcodedConfig.OnnxVersions);
var onnxTasks = ResolveTasks(onnxPaths, HardcodedConfig.OnnxTasks);

var maxInputWidth = HardcodedConfig.PipelineMaxInputWidth;
var maxInputHeight = HardcodedConfig.PipelineMaxInputHeight;

var hub = new ModelHub();
var models = new List<YoloModelBase>();
var pipelines = new List<PipelineTest>();
var modelLoads = new List<ModelLoadInfo>();

try
{
    Console.WriteLine($"Engines: {enginePaths.Count}");
    foreach (var path in enginePaths)
        Console.WriteLine(path);
    Console.WriteLine($"ONNX Models: {onnxPaths.Count}");
    foreach (var path in onnxPaths)
        Console.WriteLine(path);

    Console.WriteLine($"Images: {imageInputs.Count}");
    foreach (var img in imageInputs)
        Console.WriteLine(img.Path);
    Console.WriteLine($"CPU threads: {cpuThreads}");

    var totalLoadMs = 0.0;
    for (var i = 0; i < enginePaths.Count; i++)
    {
        var enginePath = enginePaths[i];
        var version = versions[i];
        var task = tasks[i];
        var modelId = BuildModelId(enginePath, i, version, task, "gpu");

        var labels = ResolveLabels(enginePath, HardcodedConfig.LabelsPath);
        var config = new YoloConfig
        {
            Version = version,
            Task = task,
            InputWidth = inputWidth,
            InputHeight = inputHeight,
            InputChannels = inputChannels,
            ConfidenceThreshold = conf,
            IouThreshold = iou,
            MaxDetections = maxDet,
            ClassNames = labels,
            Backend = InferenceBackend.TensorRT,
        };
        YoloVersionAdapter.ApplyVersionDefaults(config);

        var loadSw = Stopwatch.StartNew();
        var model = CreateModel(version, modelId, enginePath, config, deviceId, hub);
        loadSw.Stop();
        var loadMs = loadSw.Elapsed.TotalMilliseconds;
        totalLoadMs += loadMs;
        modelLoads.Add(new ModelLoadInfo(modelId, enginePath, loadMs));
        Console.WriteLine($"Model loaded: {modelId} -> {loadMs:F2} ms");
        models.Add(model);

        var fastOptions = new YoloPipelineOptions
        {
            BatchSize = 1,
            WorkerCount = 1,
            MaxBatchDelayMs = 2,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
            Device = InferenceDevice.Gpu,
        };

        var throughputOptions = new YoloPipelineOptions
        {
            BatchSize = 4,
            WorkerCount = 2,
            MaxBatchDelayMs = 10,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
            Device = InferenceDevice.Gpu,
        };

        pipelines.Add(new PipelineTest(model, "gpu-fast", model.CreatePipeline("gpu-fast", fastOptions), "gpu", "tensorrt", "engine"));
        pipelines.Add(new PipelineTest(model, "gpu-throughput", model.CreatePipeline("gpu-throughput", throughputOptions), "gpu", "tensorrt", "engine"));
    }

    for (var i = 0; i < onnxPaths.Count; i++)
    {
        var onnxPath = onnxPaths[i];
        var version = onnxVersions[i];
        var task = onnxTasks[i];
        var modelId = BuildModelId(onnxPath, i, version, task, "cpu");

        var labels = ResolveLabels(onnxPath, HardcodedConfig.LabelsPath);
        var config = new YoloConfig
        {
            Version = version,
            Task = task,
            InputWidth = inputWidth,
            InputHeight = inputHeight,
            InputChannels = inputChannels,
            ConfidenceThreshold = conf,
            IouThreshold = iou,
            MaxDetections = maxDet,
            ClassNames = labels,
            Backend = InferenceBackend.OnnxRuntime,
        };
        YoloVersionAdapter.ApplyVersionDefaults(config);

        var loadSw = Stopwatch.StartNew();
        var model = CreateModel(version, modelId, onnxPath, config, deviceId, hub);
        loadSw.Stop();
        var loadMs = loadSw.Elapsed.TotalMilliseconds;
        totalLoadMs += loadMs;
        modelLoads.Add(new ModelLoadInfo(modelId, onnxPath, loadMs));
        Console.WriteLine($"Model loaded: {modelId} -> {loadMs:F2} ms");
        models.Add(model);

        var fastOptions = new YoloPipelineOptions
        {
            BatchSize = 1,
            WorkerCount = 1,
            MaxBatchDelayMs = 2,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
            Device = InferenceDevice.Cpu,
            CpuThreads = cpuThreads,
        };

        var throughputOptions = new YoloPipelineOptions
        {
            BatchSize = 4,
            WorkerCount = 2,
            MaxBatchDelayMs = 10,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
            Device = InferenceDevice.Cpu,
            CpuThreads = cpuThreads,
        };

        pipelines.Add(new PipelineTest(model, "cpu-fast", model.CreatePipeline("cpu-fast", fastOptions), "cpu", "onnxruntime", "onnx"));
        pipelines.Add(new PipelineTest(model, "cpu-throughput", model.CreatePipeline("cpu-throughput", throughputOptions), "cpu", "onnxruntime", "onnx"));
    }

    Console.WriteLine();
    Console.WriteLine($"Pipelines: {pipelines.Count}");
    Console.WriteLine($"Total model load time: {totalLoadMs:F2} ms");
    Console.WriteLine();

    var perPipelineRuns = HardcodedConfig.PipelineIterations;
    if (perPipelineRuns < 1)
        perPipelineRuns = 1;

    if (HardcodedConfig.RunSequentialBenchmark)
    {
        Console.WriteLine();
        Console.WriteLine("=== Sequential Pipeline Benchmark ===");
        var sequentialResults = new List<PipelineSummary>();
        foreach (var pipeline in pipelines)
            sequentialResults.Add(RunPipeline(pipeline, imageInputs, perPipelineRuns, "sequential"));
        PrintSummaries(sequentialResults);
    }

    if (HardcodedConfig.RunParallelBenchmark)
    {
        Console.WriteLine();
        Console.WriteLine("=== Parallel Pipeline Benchmark (multi-model pipelines) ===");
        var parallelSw = Stopwatch.StartNew();
        var parallelTasks = pipelines
            .Select(p => Task.Run(() => RunPipeline(p, imageInputs, perPipelineRuns, "parallel")))
            .ToArray();
        var parallelResults = await Task.WhenAll(parallelTasks);
        parallelSw.Stop();
        PrintSummaries(parallelResults);
        Console.WriteLine($"Parallel total wall time: {parallelSw.Elapsed.TotalMilliseconds:F2} ms");
    }
}
finally
{
    foreach (var model in models)
        model.Dispose();
    hub.Dispose();
}

static async Task RunLegacyAsync()
{
    Console.WriteLine("Legacy mode: single model + single pipeline (same defaults as the original example).");

    EnsureCudaBinsOnPath();

    var enginePath = HardcodedConfig.LegacyEnginePath;

    if (!File.Exists(enginePath))
    {
        Console.WriteLine($"Engine not found: {enginePath}");
        return;
    }

    var modelDir = Path.GetDirectoryName(enginePath) ?? Directory.GetCurrentDirectory();

    var imagePath = HardcodedConfig.LegacyImagePath;
    if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
    {
        var exts = new[] { ".jpg", ".jpeg", ".png", ".bmp" };
        imagePath = Directory.EnumerateFiles(modelDir)
            .FirstOrDefault(p => exts.Contains(Path.GetExtension(p), StringComparer.OrdinalIgnoreCase));
    }

    if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
    {
        Console.WriteLine($"Image not found in: {modelDir}");
        return;
    }

    var labelsPath = HardcodedConfig.LabelsPath
        ?? Path.Combine(modelDir, "labels.txt");

    var classNames = File.Exists(labelsPath)
        ? YoloLabels.LoadFromFile(labelsPath)
        : Array.Empty<string>();

    var deviceId = HardcodedConfig.DeviceId;
    var inputWidth = HardcodedConfig.InputWidth;
    var inputHeight = HardcodedConfig.InputHeight;
    var inputChannels = HardcodedConfig.InputChannels;
    var conf = HardcodedConfig.ConfidenceThreshold;
    var iou = HardcodedConfig.IouThreshold;
    var maxDet = HardcodedConfig.LegacyMaxDetections;

    var config = new YoloConfig
    {
        Version = YoloVersion.V8,
        Task = YoloTask.Detect,
        InputWidth = inputWidth,
        InputHeight = inputHeight,
        InputChannels = inputChannels,
        ConfidenceThreshold = conf,
        IouThreshold = iou,
        MaxDetections = maxDet,
        ClassNames = classNames,
    };
    YoloVersionAdapter.ApplyVersionDefaults(config);

    var loadSw = Stopwatch.StartNew();
    using var model = new YoloV8Model("yolo-v8", enginePath, config, deviceId: deviceId);
    loadSw.Stop();
    Console.WriteLine($"Model loaded: yolo-v8 -> {loadSw.Elapsed.TotalMilliseconds:F2} ms");
    using var pipeline = new YoloGpuThroughputPipeline(model, new YoloGpuThroughputOptions
    {
        MaxConcurrency = 2,
        PipelineOptions = new YoloPipelineOptions
        {
            BatchSize = 1,
            WorkerCount = 1,
            MaxInputWidth = 4096,
            MaxInputHeight = 4096,
        },
    });

    var image = YoloEncodedImage.FromFile(imagePath);
    var warmup = HardcodedConfig.LegacyWarmup;
    var iterations = HardcodedConfig.LegacyIterations;
    if (iterations < 1)
        iterations = 1;

    Console.WriteLine($"Engine: {enginePath}");
    Console.WriteLine($"Image: {imagePath}");
    Console.WriteLine($"Warmup: {warmup}, Iterations: {iterations}");

    for (var i = 0; i < warmup; i++)
    {
        var _ = await pipeline.EnqueueAsync(image, "warmup", i).ConfigureAwait(false);
    }

    var totalMs = 0.0;
    for (var i = 0; i < iterations; i++)
    {
        var sw = Stopwatch.StartNew();
        var result = await pipeline.EnqueueAsync(image, "demo", i).ConfigureAwait(false);
        sw.Stop();
        totalMs += sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Iter {i + 1}/{iterations}: {sw.Elapsed.TotalMilliseconds:F2} ms, detections={result.Detections.Count}");
    }

    var avg = totalMs / iterations;
    Console.WriteLine($"Average: {avg:F2} ms");
}

static List<string> ResolveEnginePaths(IEnumerable<string> hardcoded)
{
    return ExpandModelCandidates(hardcoded, ".engine");
}

static List<string> ResolveOnnxPaths(IEnumerable<string> hardcoded)
{
    return ExpandModelCandidates(hardcoded, ".onnx");
}

static List<ImageInput> LoadImages(IReadOnlyList<string> enginePaths, IEnumerable<string> hardcoded)
{
    var results = ExpandImageCandidates(hardcoded);
    if (results.Count == 0)
    {
        var firstDir = Path.GetDirectoryName(enginePaths[0]);
        if (!string.IsNullOrWhiteSpace(firstDir))
            results = ExpandImageCandidates(new[] { firstDir });
    }

    var images = new List<ImageInput>();
    foreach (var path in results)
    {
        try
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length == 0)
                continue;
            images.Add(new ImageInput(path, bytes));
        }
        catch
        {
            // Skip unreadable files in demo mode.
        }
    }

    return images;
}

static List<string> ExpandModelCandidates(IEnumerable<string> candidates, string extension)
{
    var results = new List<string>();
    foreach (var candidate in candidates)
    {
        if (string.IsNullOrWhiteSpace(candidate))
            continue;

        var expanded = Environment.ExpandEnvironmentVariables(candidate.Trim());
        if (Directory.Exists(expanded))
        {
            results.AddRange(Directory.EnumerateFiles(expanded, $"*{extension}", SearchOption.TopDirectoryOnly));
            continue;
        }

        if (File.Exists(expanded))
        {
            results.Add(expanded);
            continue;
        }

        if (HasWildcard(expanded))
        {
            var dir = Path.GetDirectoryName(expanded);
            if (string.IsNullOrWhiteSpace(dir))
                dir = Directory.GetCurrentDirectory();
            var pattern = Path.GetFileName(expanded);
            if (!string.IsNullOrWhiteSpace(pattern) && Directory.Exists(dir))
                results.AddRange(Directory.EnumerateFiles(dir, pattern, SearchOption.TopDirectoryOnly));
        }
    }

    return results
        .Where(File.Exists)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToList();
}

static List<string> ExpandImageCandidates(IEnumerable<string> candidates)
{
    var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp"
    };

    var results = new List<string>();
    foreach (var candidate in candidates)
    {
        if (string.IsNullOrWhiteSpace(candidate))
            continue;

        var expanded = Environment.ExpandEnvironmentVariables(candidate.Trim());
        if (Directory.Exists(expanded))
        {
            results.AddRange(Directory.EnumerateFiles(expanded)
                .Where(p => exts.Contains(Path.GetExtension(p))));
            continue;
        }

        if (File.Exists(expanded))
        {
            results.Add(expanded);
            continue;
        }

        if (HasWildcard(expanded))
        {
            var dir = Path.GetDirectoryName(expanded);
            if (string.IsNullOrWhiteSpace(dir))
                dir = Directory.GetCurrentDirectory();
            var pattern = Path.GetFileName(expanded);
            if (!string.IsNullOrWhiteSpace(pattern) && Directory.Exists(dir))
                results.AddRange(Directory.EnumerateFiles(dir, pattern, SearchOption.TopDirectoryOnly));
        }
    }

    return results
        .Where(File.Exists)
        .Distinct(StringComparer.OrdinalIgnoreCase)
        .ToList();
}

static bool HasWildcard(string path)
{
    return path.Contains('*') || path.Contains('?');
}

static List<YoloVersion> ResolveVersions(IReadOnlyList<string> enginePaths, IReadOnlyList<YoloVersion> hardcoded)
{
    var results = new List<YoloVersion>();
    for (var i = 0; i < enginePaths.Count; i++)
    {
        if (i < hardcoded.Count)
        {
            results.Add(hardcoded[i]);
            continue;
        }

        var inferred = TryInferVersionFromPath(enginePaths[i]);
        results.Add(inferred ?? YoloVersion.V8);
    }

    return results;
}

static List<YoloTask> ResolveTasks(IReadOnlyList<string> enginePaths, IReadOnlyList<YoloTask> hardcoded)
{
    var results = new List<YoloTask>();
    for (var i = 0; i < enginePaths.Count; i++)
    {
        if (i < hardcoded.Count)
        {
            results.Add(hardcoded[i]);
            continue;
        }
        results.Add(YoloTask.Detect);
    }

    return results;
}

static YoloVersion? TryInferVersionFromPath(string path)
{
    var name = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
    var map = new Dictionary<string, YoloVersion>
    {
        { "v11", YoloVersion.V11 },
        { "v10", YoloVersion.V10 },
        { "v9", YoloVersion.V9 },
        { "v8", YoloVersion.V8 },
        { "v7", YoloVersion.V7 },
        { "v6", YoloVersion.V6 },
        { "v5", YoloVersion.V5 },
        { "v4", YoloVersion.V4 },
        { "v3", YoloVersion.V3 },
    };

    foreach (var pair in map)
    {
        if (name.Contains(pair.Key))
            return pair.Value;
    }

    return null;
}

static int ResolveCpuThreads(string[] args)
{
    const string prefix = "--cpu-threads=";
    foreach (var arg in args)
    {
        if (!arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            continue;
        var raw = arg.Substring(prefix.Length).Trim();
        if (int.TryParse(raw, out var value) && value > 0)
            return value;
    }

    var env = Environment.GetEnvironmentVariable("CUDARS_CPU_THREADS");
    if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out var fromEnv) && fromEnv > 0)
        return fromEnv;

    return HardcodedConfig.CpuThreads;
}

static string[] ResolveLabels(string enginePath, string? overridePath)
{
    if (!string.IsNullOrWhiteSpace(overridePath) && File.Exists(overridePath))
        return YoloLabels.LoadFromFile(overridePath);

    var dir = Path.GetDirectoryName(enginePath);
    if (string.IsNullOrWhiteSpace(dir))
        return Array.Empty<string>();

    var labelsPath = Path.Combine(dir, "labels.txt");
    if (!File.Exists(labelsPath))
        return Array.Empty<string>();

    return YoloLabels.LoadFromFile(labelsPath);
}

static YoloModelBase CreateModel(
    YoloVersion version,
    string modelId,
    string enginePath,
    YoloConfig config,
    int deviceId,
    ModelHub hub)
{
    return version switch
    {
        YoloVersion.V3 => new YoloV3Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V4 => new YoloV4Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V5 => new YoloV5Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V6 => new YoloV6Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V7 => new YoloV7Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V8 => new YoloV8Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V9 => new YoloV9Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V10 => new YoloV10Model(modelId, enginePath, config, deviceId, hub),
        YoloVersion.V11 => new YoloV11Model(modelId, enginePath, config, deviceId, hub),
        _ => new YoloV8Model(modelId, enginePath, config, deviceId, hub),
    };
}

static string BuildModelId(string modelPath, int index, YoloVersion version, YoloTask task, string deviceTag)
{
    var name = Path.GetFileNameWithoutExtension(modelPath);
    if (string.IsNullOrWhiteSpace(name))
        name = $"model-{index}";
    return $"{name}-{deviceTag}-{version}-{task}".ToLowerInvariant();
}

static PipelineSummary RunPipeline(
    PipelineTest pipeline,
    IReadOnlyList<ImageInput> images,
    int iterations,
    string tag)
{
    Console.WriteLine();
    Console.WriteLine($"[Pipeline:{tag}] model={pipeline.Model.ModelId} pipeline={pipeline.PipelineId} device={pipeline.DeviceTag} backend={pipeline.BackendTag} source={pipeline.SourceTag}");

    var summary = new PipelineSummary(
        pipeline.Model.ModelId,
        pipeline.PipelineId,
        pipeline.DeviceTag,
        pipeline.BackendTag,
        pipeline.SourceTag,
        iterations);

    var totalSw = Stopwatch.StartNew();
    double firstMs = 0;
    double steadyTotal = 0;
    var steadyCount = 0;
    for (var i = 0; i < iterations; i++)
    {
        var image = images[i % images.Count];
        var sw = Stopwatch.StartNew();
        try
        {
            var result = pipeline.Pipeline.Run(image.Bytes, pipeline.PipelineId, i);
            sw.Stop();
            var elapsedMs = sw.Elapsed.TotalMilliseconds;
            summary.TotalMs += elapsedMs;
            if (i == 0)
                firstMs = elapsedMs;
            else
            {
                steadyTotal += elapsedMs;
                steadyCount++;
            }
            summary.TotalDetections += result.TotalCount;
            summary.SuccessCount += result.Success ? 1 : 0;
            summary.FailureCount += result.Success ? 0 : 1;
            Console.WriteLine($"OK {Path.GetFileName(image.Path)} -> {elapsedMs:F2} ms, detections={result.TotalCount}");
        }
        catch (Exception ex)
        {
            sw.Stop();
            var elapsedMs = sw.Elapsed.TotalMilliseconds;
            summary.TotalMs += elapsedMs;
            if (i == 0)
                firstMs = elapsedMs;
            else
            {
                steadyTotal += elapsedMs;
                steadyCount++;
            }
            summary.FailureCount += 1;
            Console.WriteLine($"FAIL {Path.GetFileName(image.Path)} -> {elapsedMs:F2} ms, {ex.Message}");
        }
    }
    totalSw.Stop();

    var avg = summary.TotalMs / Math.Max(1, iterations);
    var steadyAvg = steadyTotal / Math.Max(1, steadyCount);
    summary.FirstMs = firstMs;
    summary.SteadyAvgMs = steadyAvg;
    Console.WriteLine($"Total: {summary.TotalMs:F2} ms, Avg: {avg:F2} ms, First: {firstMs:F2} ms, SteadyAvg: {steadyAvg:F2} ms, Wall: {totalSw.Elapsed.TotalMilliseconds:F2} ms");

    return summary;
}

static void PrintSummaries(IEnumerable<PipelineSummary> summaries)
{
    PrintSummariesByDevice(summaries, "gpu", "=== GPU Summary ===");
    PrintSummariesByDevice(summaries, "cpu", "=== CPU Summary ===");
}

static void PrintSummariesByDevice(IEnumerable<PipelineSummary> summaries, string deviceTag, string title)
{
    var list = summaries
        .Where(s => string.Equals(s.DeviceTag, deviceTag, StringComparison.OrdinalIgnoreCase))
        .ToList();
    if (list.Count == 0)
        return;

    Console.WriteLine();
    Console.WriteLine(title);
    foreach (var summary in list)
    {
        var avg = summary.TotalMs / Math.Max(1, summary.Iterations);
        Console.WriteLine(
            $"{summary.ModelId} | {summary.PipelineId} | source={summary.SourceTag} backend={summary.BackendTag} | runs={summary.Iterations} ok={summary.SuccessCount} fail={summary.FailureCount} det={summary.TotalDetections} totalMs={summary.TotalMs:F2} avgMs={avg:F2} firstMs={summary.FirstMs:F2} steadyAvgMs={summary.SteadyAvgMs:F2}");
    }
}

static void EnsureCudaBinsOnPath()
{
    var candidates = new List<string>();

    if (!string.IsNullOrWhiteSpace(HardcodedConfig.CudaRoot))
        candidates.Add(Path.Combine(HardcodedConfig.CudaRoot, "bin"));
    if (!string.IsNullOrWhiteSpace(HardcodedConfig.TensorRtRoot))
    {
        candidates.Add(Path.Combine(HardcodedConfig.TensorRtRoot, "bin"));
        candidates.Add(Path.Combine(HardcodedConfig.TensorRtRoot, "lib"));
    }

    var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
    var parts = new HashSet<string>(
        path.Split(';', StringSplitOptions.RemoveEmptyEntries),
        StringComparer.OrdinalIgnoreCase);

    var updated = false;
    foreach (var candidate in candidates)
    {
        if (string.IsNullOrWhiteSpace(candidate))
            continue;
        if (!Directory.Exists(candidate))
            continue;
        if (parts.Add(candidate))
            updated = true;
    }

    if (updated)
    {
        var newPath = string.Join(";", parts);
        Environment.SetEnvironmentVariable("PATH", newPath);
        Console.WriteLine("Updated PATH with CUDA/TensorRT bin directories.");
    }
}

sealed class PipelineTest
{
    public PipelineTest(YoloModelBase model, string pipelineId, YoloPipeline pipeline, string deviceTag, string backendTag, string sourceTag)
    {
        Model = model;
        PipelineId = pipelineId;
        Pipeline = pipeline;
        DeviceTag = deviceTag;
        BackendTag = backendTag;
        SourceTag = sourceTag;
    }

    public YoloModelBase Model { get; }
    public string PipelineId { get; }
    public YoloPipeline Pipeline { get; }
    public string DeviceTag { get; }
    public string BackendTag { get; }
    public string SourceTag { get; }
}

sealed class PipelineSummary
{
    public PipelineSummary(string modelId, string pipelineId, string deviceTag, string backendTag, string sourceTag, int iterations)
    {
        ModelId = modelId;
        PipelineId = pipelineId;
        DeviceTag = deviceTag;
        BackendTag = backendTag;
        SourceTag = sourceTag;
        Iterations = iterations;
    }

    public string ModelId { get; }
    public string PipelineId { get; }
    public string DeviceTag { get; }
    public string BackendTag { get; }
    public string SourceTag { get; }
    public int Iterations { get; }
    public int SuccessCount { get; set; }
    public int FailureCount { get; set; }
    public int TotalDetections { get; set; }
    public double TotalMs { get; set; }
    public double FirstMs { get; set; }
    public double SteadyAvgMs { get; set; }
}

sealed class ImageInput
{
    public ImageInput(string path, byte[] bytes)
    {
        Path = path;
        Bytes = bytes;
    }

    public string Path { get; }
    public byte[] Bytes { get; }
}

sealed class ModelLoadInfo
{
    public ModelLoadInfo(string modelId, string path, double loadMs)
    {
        ModelId = modelId;
        Path = path;
        LoadMs = loadMs;
    }

    public string ModelId { get; }
    public string Path { get; }
    public double LoadMs { get; }
}

static class HardcodedConfig
{
    public static readonly bool UseLegacyMode = false;
    public static readonly bool RunSequentialBenchmark = true;
    public static readonly bool RunParallelBenchmark = true;

    public const string CudaRoot = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6";
    public const string TensorRtRoot = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6";

    public const int DeviceId = 0;
    public const int InputWidth = 640;
    public const int InputHeight = 640;
    public const int InputChannels = 3;
    public const float ConfidenceThreshold = 0.25f;
    public const float IouThreshold = 0.45f;
    public const int MaxDetections = 300;

    public const int PipelineMaxInputWidth = 4096;
    public const int PipelineMaxInputHeight = 4096;
    public const int PipelineIterations = 10;

    public const int LegacyWarmup = 0;
    public const int LegacyIterations = 1;
    public const int LegacyMaxDetections = 100;
    public const int CpuThreads = 8;

    public const string LegacyEnginePath = @"E:\codeding\AI\onnx\best\best.engine";
    public const string LegacyImagePath = @"E:\codeding\AI\onnx\best\train_batch0.jpg";

    public static readonly string[] EnginePaths =
    {
        @"E:\codeding\AI\onnx\best\best.engine",
        // Add more TensorRT engine paths here.
    };

    public static readonly string[] OnnxModelPaths =
    {
        @"E:\codeding\AI\onnx\best\best.onnx",
        // Add more ONNX model paths here.
    };

    public static readonly string[] ImagePaths =
    {
        @"E:\codeding\AI\onnx\best\train_batch0.jpg",
        // Add more image paths here.
    };

    public static readonly YoloVersion[] Versions =
    {
        YoloVersion.V8,
        // Add more versions to match EnginePaths.
    };

    public static readonly YoloVersion[] OnnxVersions =
    {
        YoloVersion.V8,
        // Add more versions to match OnnxModelPaths.
    };

    public static readonly YoloTask[] Tasks =
    {
        YoloTask.Detect,
        // Add more tasks to match EnginePaths.
    };

    public static readonly YoloTask[] OnnxTasks =
    {
        YoloTask.Detect,
        // Add more tasks to match OnnxModelPaths.
    };

    public static readonly string? LabelsPath = null;
}

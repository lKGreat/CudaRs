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

var mode = (Environment.GetEnvironmentVariable("YOLO_MODE") ?? string.Empty).Trim();
if (string.Equals(mode, "legacy", StringComparison.OrdinalIgnoreCase) || ReadBoolEnv("YOLO_LEGACY", false))
{
    await RunLegacyAsync();
    return;
}

var enginePaths = ResolveEnginePaths();
if (enginePaths.Count == 0)
{
    Console.WriteLine("No TensorRT engines found. Set YOLO_ENGINES or YOLO_ENGINE, or place *.engine in YOLO_ENGINE_DIR.");
    return;
}

var imageInputs = LoadImages(enginePaths);
if (imageInputs.Count == 0)
{
    Console.WriteLine("No input images found. Set YOLO_IMAGES/YOLO_IMAGE or place images next to your engines.");
    return;
}

var deviceId = ReadIntEnv("YOLO_DEVICE", 0);
var inputWidth = ReadIntEnv("YOLO_INPUT_WIDTH", 640);
var inputHeight = ReadIntEnv("YOLO_INPUT_HEIGHT", 640);
var inputChannels = ReadIntEnv("YOLO_INPUT_CHANNELS", 3);
var conf = ReadFloatEnv("YOLO_CONF", 0.25f);
var iou = ReadFloatEnv("YOLO_IOU", 0.45f);
var maxDet = ReadIntEnv("YOLO_MAX_DET", 300);

var versions = ResolveVersions(enginePaths);
var tasks = ResolveTasks(enginePaths);

var maxInputWidth = ReadIntEnv("YOLO_PIPELINE_MAX_W", 4096);
var maxInputHeight = ReadIntEnv("YOLO_PIPELINE_MAX_H", 4096);

var hub = new ModelHub();
var models = new List<YoloModelBase>();
var pipelines = new List<PipelineTest>();

try
{
    Console.WriteLine($"Engines: {enginePaths.Count}");
    foreach (var path in enginePaths)
        Console.WriteLine(path);

    Console.WriteLine($"Images: {imageInputs.Count}");
    foreach (var img in imageInputs)
        Console.WriteLine(img.Path);

    for (var i = 0; i < enginePaths.Count; i++)
    {
        var enginePath = enginePaths[i];
        var version = versions[i];
        var task = tasks[i];
        var modelId = BuildModelId(enginePath, i, version, task);

        var labels = ResolveLabels(enginePath);
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
        };
        YoloVersionAdapter.ApplyVersionDefaults(config);

        var model = CreateModel(version, modelId, enginePath, config, deviceId, hub);
        models.Add(model);

        var fastOptions = new YoloPipelineOptions
        {
            BatchSize = 1,
            WorkerCount = 1,
            MaxBatchDelayMs = 2,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
        };

        var throughputOptions = new YoloPipelineOptions
        {
            BatchSize = 4,
            WorkerCount = 2,
            MaxBatchDelayMs = 10,
            AllowPartialBatch = true,
            MaxInputWidth = maxInputWidth,
            MaxInputHeight = maxInputHeight,
        };

        pipelines.Add(new PipelineTest(model, "fast", model.CreatePipeline("fast", fastOptions)));
        pipelines.Add(new PipelineTest(model, "throughput", model.CreatePipeline("throughput", throughputOptions)));
    }

    Console.WriteLine();
    Console.WriteLine($"Pipelines: {pipelines.Count}");

    var parallel = ReadBoolEnv("YOLO_PARALLEL", false);
    if (parallel)
    {
        var tasksList = pipelines
            .Select(p => Task.Run(() => RunPipeline(p, imageInputs)))
            .ToArray();
        var results = await Task.WhenAll(tasksList);
        PrintSummaries(results);
    }
    else
    {
        var results = new List<PipelineSummary>();
        foreach (var pipeline in pipelines)
            results.Add(RunPipeline(pipeline, imageInputs));
        PrintSummaries(results);
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

    var enginePath = Environment.GetEnvironmentVariable("YOLO_ENGINE")
        ?? @"E:\\codeding\\AI\\onnx\\best\\best.engine";

    if (!File.Exists(enginePath))
    {
        Console.WriteLine($"Engine not found: {enginePath}");
        return;
    }

    var modelDir = Path.GetDirectoryName(enginePath) ?? Directory.GetCurrentDirectory();

    var imagePath = Environment.GetEnvironmentVariable("YOLO_IMAGE");
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

    var labelsPath = Environment.GetEnvironmentVariable("YOLO_LABELS")
        ?? Path.Combine(modelDir, "labels.txt");

    var classNames = File.Exists(labelsPath)
        ? YoloLabels.LoadFromFile(labelsPath)
        : Array.Empty<string>();

    var deviceId = ReadIntEnv("YOLO_DEVICE", 0);
    var inputWidth = ReadIntEnv("YOLO_INPUT_WIDTH", 640);
    var inputHeight = ReadIntEnv("YOLO_INPUT_HEIGHT", 640);
    var inputChannels = ReadIntEnv("YOLO_INPUT_CHANNELS", 3);
    var conf = ReadFloatEnv("YOLO_CONF", 0.25f);
    var iou = ReadFloatEnv("YOLO_IOU", 0.45f);
    var maxDet = ReadIntEnv("YOLO_MAX_DET", 100);

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

    using var model = new YoloV8Model("yolo-v8", enginePath, config, deviceId: deviceId);
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
    var warmup = ReadIntEnv("YOLO_WARMUP", 0);
    var iterations = ReadIntEnv("YOLO_ITERATIONS", 1);
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

static List<string> ResolveEnginePaths()
{
    var candidates = new List<string>();
    candidates.AddRange(SplitEnvList("YOLO_ENGINES"));
    candidates.AddRange(SplitEnvList("YOLO_ENGINE_LIST"));
    candidates.AddRange(SplitEnvList("YOLO_ENGINE"));

    var dirCandidates = new List<string>();
    dirCandidates.AddRange(SplitEnvList("YOLO_ENGINE_DIR"));
    dirCandidates.AddRange(SplitEnvList("YOLO_ENGINE_DIRS"));

    var results = ExpandEngineCandidates(candidates);
    if (results.Count == 0 && dirCandidates.Count > 0)
        results = ExpandEngineCandidates(dirCandidates);

    if (results.Count == 0)
        results = ExpandEngineCandidates(new[] { Directory.GetCurrentDirectory() });

    return results;
}

static List<ImageInput> LoadImages(IReadOnlyList<string> enginePaths)
{
    var candidates = new List<string>();
    candidates.AddRange(SplitEnvList("YOLO_IMAGES"));
    candidates.AddRange(SplitEnvList("YOLO_IMAGE"));
    candidates.AddRange(SplitEnvList("YOLO_IMAGE_DIR"));

    var results = ExpandImageCandidates(candidates);
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

static List<string> ExpandEngineCandidates(IEnumerable<string> candidates)
{
    var results = new List<string>();
    foreach (var candidate in candidates)
    {
        if (string.IsNullOrWhiteSpace(candidate))
            continue;

        var expanded = Environment.ExpandEnvironmentVariables(candidate.Trim());
        if (Directory.Exists(expanded))
        {
            results.AddRange(Directory.EnumerateFiles(expanded, "*.engine", SearchOption.TopDirectoryOnly));
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

static List<string> SplitEnvList(string name)
{
    var value = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrWhiteSpace(value))
        return new List<string>();

    return value
        .Split(new[] { ';', ',', '|', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
        .Select(v => v.Trim())
        .Where(v => !string.IsNullOrWhiteSpace(v))
        .ToList();
}

static List<YoloVersion> ResolveVersions(IReadOnlyList<string> enginePaths)
{
    var parsed = ParseVersionList(SplitEnvList("YOLO_VERSIONS"));
    if (parsed.Count == 0)
        parsed = ParseVersionList(SplitEnvList("YOLO_VERSION"));

    var results = new List<YoloVersion>();
    for (var i = 0; i < enginePaths.Count; i++)
    {
        if (i < parsed.Count)
        {
            results.Add(parsed[i]);
            continue;
        }

        var inferred = TryInferVersionFromPath(enginePaths[i]);
        results.Add(inferred ?? YoloVersion.V8);
    }

    return results;
}

static List<YoloTask> ResolveTasks(IReadOnlyList<string> enginePaths)
{
    var parsed = ParseTaskList(SplitEnvList("YOLO_TASKS"));
    if (parsed.Count == 0)
        parsed = ParseTaskList(SplitEnvList("YOLO_TASK"));

    var results = new List<YoloTask>();
    for (var i = 0; i < enginePaths.Count; i++)
    {
        if (i < parsed.Count)
        {
            results.Add(parsed[i]);
            continue;
        }
        results.Add(YoloTask.Detect);
    }

    return results;
}

static List<YoloVersion> ParseVersionList(IReadOnlyList<string> tokens)
{
    var results = new List<YoloVersion>();
    foreach (var token in tokens)
    {
        var version = ParseVersion(token);
        if (version != null)
            results.Add(version.Value);
    }
    return results;
}

static List<YoloTask> ParseTaskList(IReadOnlyList<string> tokens)
{
    var results = new List<YoloTask>();
    foreach (var token in tokens)
    {
        if (Enum.TryParse<YoloTask>(token, true, out var task))
            results.Add(task);
    }
    return results;
}

static YoloVersion? ParseVersion(string token)
{
    if (string.IsNullOrWhiteSpace(token))
        return null;

    var trimmed = token.Trim();
    if (Enum.TryParse<YoloVersion>(trimmed, true, out var enumVersion))
        return enumVersion;

    var lower = trimmed.ToLowerInvariant();
    if (lower.StartsWith("yolov"))
        lower = lower.Substring(5);
    if (lower.StartsWith("v"))
        lower = lower.Substring(1);

    if (int.TryParse(lower, out var num))
    {
        return num switch
        {
            3 => YoloVersion.V3,
            4 => YoloVersion.V4,
            5 => YoloVersion.V5,
            6 => YoloVersion.V6,
            7 => YoloVersion.V7,
            8 => YoloVersion.V8,
            9 => YoloVersion.V9,
            10 => YoloVersion.V10,
            11 => YoloVersion.V11,
            _ => null
        };
    }

    return null;
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

static string[] ResolveLabels(string enginePath)
{
    var overridePath = Environment.GetEnvironmentVariable("YOLO_LABELS");
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

static string BuildModelId(string enginePath, int index, YoloVersion version, YoloTask task)
{
    var name = Path.GetFileNameWithoutExtension(enginePath);
    if (string.IsNullOrWhiteSpace(name))
        name = $"engine-{index}";
    return $"{name}-{version}-{task}".ToLowerInvariant();
}

static PipelineSummary RunPipeline(PipelineTest pipeline, IReadOnlyList<ImageInput> images)
{
    Console.WriteLine();
    Console.WriteLine($"[Pipeline] model={pipeline.Model.ModelId} pipeline={pipeline.PipelineId}");

    var summary = new PipelineSummary(pipeline.Model.ModelId, pipeline.PipelineId);

    for (var i = 0; i < images.Count; i++)
    {
        var image = images[i];
        var sw = Stopwatch.StartNew();
        try
        {
            var result = pipeline.Pipeline.Run(image.Bytes, pipeline.PipelineId, i);
            sw.Stop();
            summary.TotalMs += sw.ElapsedMilliseconds;
            summary.TotalDetections += result.TotalCount;
            summary.SuccessCount += result.Success ? 1 : 0;
            summary.FailureCount += result.Success ? 0 : 1;
            Console.WriteLine($"OK {Path.GetFileName(image.Path)} -> {result.TotalCount} detections, {result.TotalMs:F2} ms");
        }
        catch (Exception ex)
        {
            sw.Stop();
            summary.TotalMs += sw.ElapsedMilliseconds;
            summary.FailureCount += 1;
            Console.WriteLine($"FAIL {Path.GetFileName(image.Path)} -> {ex.Message}");
        }
    }

    return summary;
}

static void PrintSummaries(IEnumerable<PipelineSummary> summaries)
{
    Console.WriteLine();
    Console.WriteLine("=== Summary ===");
    foreach (var summary in summaries)
    {
        Console.WriteLine(
            $"{summary.ModelId} | {summary.PipelineId} | ok={summary.SuccessCount} fail={summary.FailureCount} det={summary.TotalDetections} wallMs={summary.TotalMs}");
    }
}

static int ReadIntEnv(string name, int fallback)
{
    var value = Environment.GetEnvironmentVariable(name);
    return int.TryParse(value, out var parsed) ? parsed : fallback;
}

static float ReadFloatEnv(string name, float fallback)
{
    var value = Environment.GetEnvironmentVariable(name);
    return float.TryParse(value, out var parsed) ? parsed : fallback;
}

static void EnsureCudaBinsOnPath()
{
    var candidates = new List<string>();

    var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
    if (!string.IsNullOrWhiteSpace(cudaPath))
        candidates.Add(Path.Combine(cudaPath.Trim(), "bin"));

    var tensorrtRoot = Environment.GetEnvironmentVariable("TENSORRT_ROOT");
    if (!string.IsNullOrWhiteSpace(tensorrtRoot))
    {
        candidates.Add(Path.Combine(tensorrtRoot.Trim(), "bin"));
        candidates.Add(Path.Combine(tensorrtRoot.Trim(), "lib"));
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

static bool ReadBoolEnv(string name, bool fallback)
{
    var value = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrWhiteSpace(value))
        return fallback;

    return value.Equals("1", StringComparison.OrdinalIgnoreCase)
        || value.Equals("true", StringComparison.OrdinalIgnoreCase)
        || value.Equals("yes", StringComparison.OrdinalIgnoreCase)
        || value.Equals("y", StringComparison.OrdinalIgnoreCase);
}

sealed class PipelineTest
{
    public PipelineTest(YoloModelBase model, string pipelineId, YoloPipeline pipeline)
    {
        Model = model;
        PipelineId = pipelineId;
        Pipeline = pipeline;
    }

    public YoloModelBase Model { get; }
    public string PipelineId { get; }
    public YoloPipeline Pipeline { get; }
}

sealed class PipelineSummary
{
    public PipelineSummary(string modelId, string pipelineId)
    {
        ModelId = modelId;
        PipelineId = pipelineId;
    }

    public string ModelId { get; }
    public string PipelineId { get; }
    public int SuccessCount { get; set; }
    public int FailureCount { get; set; }
    public int TotalDetections { get; set; }
    public long TotalMs { get; set; }
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

using System;
using System.IO;
using System.Linq;
using CudaRS.Yolo;

Console.WriteLine("=== CudaRS.Yolo Example ===");

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

var config = new YoloConfig
{
    Version = YoloVersion.V8,
    Task = YoloTask.Detect,
    InputWidth = 640,
    InputHeight = 640,
    InputChannels = 3,
    ConfidenceThreshold = 0.25f,
    IouThreshold = 0.45f,
    MaxDetections = 100,
    ClassNames = classNames,
};

YoloVersionAdapter.ApplyVersionDefaults(config);

using var model = new YoloV8Model("yolo-v8", enginePath, config, deviceId: 0);
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

var imageBytes = File.ReadAllBytes(imagePath);
var result = await pipeline.EnqueueAsync(imageBytes, "demo", 0);

Console.WriteLine($"Success: {result.Success}");
Console.WriteLine($"Detections: {result.Detections.Count}");
foreach (var det in result.Detections.Take(5))
    Console.WriteLine(det);

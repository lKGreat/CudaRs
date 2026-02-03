using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

/// <summary>
/// Single model inference result (isolated per model).
/// </summary>
public sealed class ModelInferenceResult
{
    public Guid ResultId { get; } = Guid.NewGuid();
    public string ModelId { get; init; } = string.Empty;
    public YoloVersion Version { get; init; }
    public YoloTask Task { get; init; }
    public string ChannelId { get; init; } = string.Empty;
    public long FrameIndex { get; init; }
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;

    // Timing
    public double PreprocessMs { get; init; }
    public double InferenceMs { get; init; }
    public double PostprocessMs { get; init; }
    public double TotalMs => PreprocessMs + InferenceMs + PostprocessMs;

    // Status
    public bool Success { get; init; }
    public string? ErrorMessage { get; init; }

    // Results by task type
    public IReadOnlyList<Detection> Detections { get; init; } = Array.Empty<Detection>();
    public IReadOnlyList<SegmentationDetection> Segmentations { get; init; } = Array.Empty<SegmentationDetection>();
    public IReadOnlyList<PoseDetection> Poses { get; init; } = Array.Empty<PoseDetection>();
    public IReadOnlyList<ObbDetection> ObbDetections { get; init; } = Array.Empty<ObbDetection>();
    public IReadOnlyList<Classification> Classifications { get; init; } = Array.Empty<Classification>();

    // NMS details (when applicable)
    public NmsSummary? Nms { get; init; }

    // Raw outputs
    public IReadOnlyDictionary<string, float[]>? RawOutputs { get; init; }

    // Memory stats
    public long MemoryUsedBytes { get; init; }
    public int GpuDeviceId { get; init; }

    /// <summary>Get all bounding boxes from any task type.</summary>
    public IEnumerable<BoundingBox> GetAllBoxes()
    {
        foreach (var d in Detections) yield return d.Box;
        foreach (var s in Segmentations) yield return s.Box;
        foreach (var p in Poses) yield return p.Box;
        foreach (var o in ObbDetections) yield return o.AxisAlignedBox;
    }

    /// <summary>Get total detection count across all task types.</summary>
    public int TotalCount =>
        Detections.Count + Segmentations.Count + Poses.Count + ObbDetections.Count + Classifications.Count;
}

/// <summary>
/// Summary of non-maximum suppression settings and results.
/// </summary>
public sealed class NmsSummary
{
    public float IouThreshold { get; init; }
    public int MaxDetections { get; init; }
    public bool ClassAgnostic { get; init; }
    public int PreNmsCount { get; init; }
    public int PostNmsCount { get; init; }
}

/// <summary>
/// Aggregated inference result across multiple models (isolated per model).
/// </summary>
public sealed class MultiModelInferenceResult
{
    public Guid AggregateId { get; } = Guid.NewGuid();
    public string ChannelId { get; init; } = string.Empty;
    public long FrameIndex { get; init; }
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;
    public double TotalMs { get; init; }

    public IReadOnlyDictionary<string, ModelInferenceResult> ModelResults { get; init; }
        = new Dictionary<string, ModelInferenceResult>();

    public ModelInferenceResult? GetModelResult(string modelId)
        => ModelResults.TryGetValue(modelId, out var r) ? r : null;

    public IReadOnlyList<Detection> GetDetections(string modelId)
        => GetModelResult(modelId)?.Detections ?? Array.Empty<Detection>();

    public IReadOnlyList<SegmentationDetection> GetSegmentations(string modelId)
        => GetModelResult(modelId)?.Segmentations ?? Array.Empty<SegmentationDetection>();

    public IReadOnlyList<PoseDetection> GetPoses(string modelId)
        => GetModelResult(modelId)?.Poses ?? Array.Empty<PoseDetection>();

    public IReadOnlyList<ObbDetection> GetObbDetections(string modelId)
        => GetModelResult(modelId)?.ObbDetections ?? Array.Empty<ObbDetection>();

    public IReadOnlyList<Classification> GetClassifications(string modelId)
        => GetModelResult(modelId)?.Classifications ?? Array.Empty<Classification>();

    public IEnumerable<(string ModelId, Detection Detection)> GetAllDetections()
    {
        foreach (var (modelId, result) in ModelResults)
            foreach (var det in result.Detections)
                yield return (modelId, det);
    }

    public IEnumerable<(string ModelId, SegmentationDetection Segmentation)> GetAllSegmentations()
    {
        foreach (var (modelId, result) in ModelResults)
            foreach (var seg in result.Segmentations)
                yield return (modelId, seg);
    }

    public IEnumerable<(string ModelId, PoseDetection Pose)> GetAllPoses()
    {
        foreach (var (modelId, result) in ModelResults)
            foreach (var pose in result.Poses)
                yield return (modelId, pose);
    }

    public bool AllSucceeded
    {
        get
        {
            foreach (var r in ModelResults.Values)
                if (!r.Success) return false;
            return ModelResults.Count > 0;
        }
    }

    public long TotalMemoryUsedBytes
    {
        get
        {
            long total = 0;
            foreach (var r in ModelResults.Values)
                total += r.MemoryUsedBytes;
            return total;
        }
    }
}

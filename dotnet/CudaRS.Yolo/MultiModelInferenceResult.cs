using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

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

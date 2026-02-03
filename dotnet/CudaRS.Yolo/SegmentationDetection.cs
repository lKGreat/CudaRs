namespace CudaRS.Yolo;

/// <summary>
/// Segmentation detection result.
/// </summary>
public sealed class SegmentationDetection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public SegmentationMask Mask { get; init; } = null!;
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {Box} [mask {Mask.Width}x{Mask.Height}]";
}

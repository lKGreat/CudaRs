namespace CudaRS.Yolo;

/// <summary>
/// Single object detection result.
/// </summary>
public sealed class Detection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {Box}";
}

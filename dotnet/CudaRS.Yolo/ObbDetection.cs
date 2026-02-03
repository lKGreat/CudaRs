namespace CudaRS.Yolo;

/// <summary>
/// Oriented bounding box detection result.
/// </summary>
public sealed class ObbDetection
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }
    public RotatedBox RotatedBox { get; init; }
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public BoundingBox AxisAlignedBox => RotatedBox.GetAxisAlignedBox();

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1} @ {RotatedBox}";
}

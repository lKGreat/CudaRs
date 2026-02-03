namespace CudaRS.Yolo;

/// <summary>
/// Pose detection result.
/// </summary>
public sealed class PoseDetection
{
    public int ClassId { get; init; } = 0;
    public string ClassName { get; init; } = "person";
    public float Confidence { get; init; }
    public BoundingBox Box { get; init; }
    public Pose Pose { get; init; } = null!;
    public int SourceWidth { get; init; }
    public int SourceHeight { get; init; }

    public override string ToString()
        => $"{ClassName}: {Confidence:P1} @ {Box} [{Pose.VisibleCount}/17 keypoints]";
}

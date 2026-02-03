namespace CudaRS.Yolo;

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

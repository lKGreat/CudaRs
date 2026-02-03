using System;

namespace CudaRS.Yolo;

/// <summary>
/// Configuration specific to a YOLO version.
/// </summary>
public sealed class YoloVersionConfig
{
    public YoloVersion Version { get; init; }
    public bool AnchorBased { get; init; }
    public bool HasObjectness { get; init; }
    public int[] DefaultStrides { get; init; } = Array.Empty<int>();
    public float[][] DefaultAnchors { get; init; } = Array.Empty<float[]>();
    public OutputLayout OutputOrder { get; init; }
    public BoxFormat BoxFormat { get; init; }
    public YoloTask[] SupportedTasks { get; init; } = Array.Empty<YoloTask>();
    public bool NmsBuiltIn { get; init; } = false;
}

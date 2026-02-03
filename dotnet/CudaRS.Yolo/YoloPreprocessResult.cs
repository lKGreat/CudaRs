using System;

namespace CudaRS.Yolo;

public sealed class YoloPreprocessResult
{
    public float[] Input { get; init; } = Array.Empty<float>();
    public int[] InputShape { get; init; } = Array.Empty<int>();
    public float Scale { get; init; }
    public int PadX { get; init; }
    public int PadY { get; init; }
    public int OriginalWidth { get; init; }
    public int OriginalHeight { get; init; }
}

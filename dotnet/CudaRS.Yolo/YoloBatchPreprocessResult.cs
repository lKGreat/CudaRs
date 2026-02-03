using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

public sealed class YoloBatchPreprocessResult
{
    public float[] Input { get; init; } = Array.Empty<float>();
    public int[] InputShape { get; init; } = Array.Empty<int>();
    public IReadOnlyList<YoloPreprocessResult> Items { get; init; } = Array.Empty<YoloPreprocessResult>();
}

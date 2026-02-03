using System;

namespace CudaRS.Yolo;

public sealed class TensorOutput
{
    public string Name { get; init; } = string.Empty;
    public float[] Data { get; init; } = Array.Empty<float>();
    public int[] Shape { get; init; } = Array.Empty<int>();
}

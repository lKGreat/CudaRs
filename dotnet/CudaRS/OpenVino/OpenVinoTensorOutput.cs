using System;

namespace CudaRS.OpenVino;

public sealed class OpenVinoTensorOutput
{
    public float[] Data { get; init; } = Array.Empty<float>();
    public int[] Shape { get; init; } = Array.Empty<int>();
}

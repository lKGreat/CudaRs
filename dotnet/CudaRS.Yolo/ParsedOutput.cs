using System;

namespace CudaRS.Yolo;

/// <summary>
/// Parsed output tensor information.
/// </summary>
public sealed class ParsedOutput
{
    public float[] Data { get; init; } = Array.Empty<float>();
    public int Batch { get; init; }
    public int Channels { get; init; }
    public int Count { get; init; }
    public bool Transposed { get; init; }
    public bool HasObjectness { get; init; }
    public int NumClasses { get; init; }
}

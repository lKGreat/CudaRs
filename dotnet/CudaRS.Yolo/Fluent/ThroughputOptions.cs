namespace CudaRS.Fluent;

/// <summary>
/// High-throughput configuration shared across backends.
/// </summary>
public sealed class ThroughputOptions
{
    public bool Enable { get; set; }
    public int BatchSize { get; set; } = 1;
    public int NumStreams { get; set; } = 1;
    public int MaxBatchDelayMs { get; set; } = 2;
}

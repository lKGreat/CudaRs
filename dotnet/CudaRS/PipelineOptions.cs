namespace CudaRS;

public sealed class PipelineOptions
{
    public string PipelineId { get; set; } = "default";
    public PipelineKind Kind { get; set; } = PipelineKind.Unknown;
    public string ConfigJson { get; set; } = "{}";
}

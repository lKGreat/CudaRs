namespace CudaRS;

public sealed class ChannelOptions
{
    public string Name { get; internal set; } = string.Empty;
    public string PipelineId { get; internal set; } = "default";
}

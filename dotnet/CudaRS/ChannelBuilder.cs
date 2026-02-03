using System;

namespace CudaRS;

public sealed class ChannelBuilder
{
    private readonly ChannelOptions _options;

    internal ChannelBuilder(ChannelOptions options)
    {
        _options = options;
    }

    public ChannelBuilder WithPipeline(string pipelineId)
    {
        _options.PipelineId = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId.Trim();
        return this;
    }
}

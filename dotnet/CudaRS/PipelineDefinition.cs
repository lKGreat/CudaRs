using System.Collections.Generic;

namespace CudaRS;

public sealed class PipelineDefinition
{
    public string Name { get; internal set; } = "DefaultPipeline";
    public IReadOnlyDictionary<string, ChannelOptions> Channels { get; internal set; } =
        new Dictionary<string, ChannelOptions>();
    public IReadOnlyDictionary<string, ModelOptions> Models { get; internal set; } =
        new Dictionary<string, ModelOptions>();
    public ExecutionOptions Execution { get; internal set; } = new ExecutionOptions();
}

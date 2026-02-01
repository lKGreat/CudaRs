using System;
using System.Collections.Generic;

namespace CudaRS;

public enum SceneLevel
{
    L0 = 0,
    L1 = 1,
    L2 = 2,
    L3 = 3,
}

public enum ResultRoutingMode
{
    PerChannel = 0,
    Merge = 1,
}

public enum BackpressurePolicy
{
    DropLowestPriority = 0,
    ThrottleChannel = 1,
    PauseChannel = 2,
}

public enum StreamMode
{
    Sync = 0,
    Async = 1,
}

public sealed class ChannelOptions
{
    public string Name { get; internal set; } = string.Empty;
    public int[] Shape { get; internal set; } = Array.Empty<int>();
    public string DataType { get; internal set; } = "float32";
    public string Preprocess { get; internal set; } = "noop";
    public int MaxBatch { get; internal set; } = 1;
    public int BatchTimeoutMs { get; internal set; } = 0;
    public float NormalizeMean { get; internal set; } = 0f;
    public float NormalizeStd { get; internal set; } = 1f;
    public SceneLevel ScenePriority { get; internal set; } = SceneLevel.L1;
    public int Weight { get; internal set; } = 1;
    public int MaxInFlight { get; internal set; } = 2;
    public int MaxQueueDepth { get; internal set; } = 200;
    public int MinFps { get; internal set; } = 1;
    public int MaxFps { get; internal set; } = 30;
}

public sealed class ModelOptions
{
    public string ModelId { get; internal set; } = "default";
    public string ModelPath { get; internal set; } = string.Empty;
    public string Backend { get; internal set; } = "auto";
    public string Device { get; internal set; } = "auto";
    public string Precision { get; internal set; } = "auto";
    public int WorkspaceMb { get; internal set; } = 256;
}

public sealed class PipelineDefinition
{
    public string Name { get; internal set; } = "DefaultPipeline";
    public IReadOnlyDictionary<string, ChannelOptions> Channels { get; internal set; } =
        new Dictionary<string, ChannelOptions>();
    public ModelOptions Model { get; internal set; } = new ModelOptions();
    public IReadOnlyList<string> PreprocessStages { get; internal set; } = Array.Empty<string>();
    public IReadOnlyList<string> InferStages { get; internal set; } = Array.Empty<string>();
    public IReadOnlyList<string> PostprocessStages { get; internal set; } = Array.Empty<string>();
    public ResultRoutingMode RoutingMode { get; internal set; } = ResultRoutingMode.PerChannel;
    public ExecutionOptions Execution { get; internal set; } = new ExecutionOptions();
}

public sealed class PipelineInput
{
    public IReadOnlyDictionary<string, ChannelInput> Channels { get; }

    public PipelineInput(IReadOnlyDictionary<string, ChannelInput> channels)
    {
        Channels = channels ?? throw new ArgumentNullException(nameof(channels));
    }

    public static PipelineInput FromObjects(IReadOnlyDictionary<string, object> channels)
    {
        if (channels == null)
            throw new ArgumentNullException(nameof(channels));

        var mapped = new Dictionary<string, ChannelInput>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, payload) in channels)
            mapped[name] = new ChannelInput(payload);

        return new PipelineInput(mapped);
    }
}

public sealed class ChannelInput
{
    public ChannelInput(object payload)
    {
        Payload = payload ?? throw new ArgumentNullException(nameof(payload));
    }

    public object Payload { get; }
    public SceneLevel SceneLevel { get; init; } = SceneLevel.L1;
    public DateTimeOffset Timestamp { get; init; } = DateTimeOffset.UtcNow;
}

public sealed class RunResult
{
    public string PipelineName { get; internal set; } = string.Empty;
    public bool Success { get; internal set; }
    public TimeSpan Elapsed { get; internal set; }
    public IReadOnlyDictionary<string, object> PerChannelOutputs { get; internal set; } =
        new Dictionary<string, object>();
    public IReadOnlyList<string> Diagnostics { get; internal set; } = Array.Empty<string>();
}

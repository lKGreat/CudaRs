using System;

namespace CudaRS;

public sealed class ChannelBuilder
{
    private readonly ChannelOptions _options;

    internal ChannelBuilder(ChannelOptions options)
    {
        _options = options;
    }

    public ChannelBuilder WithShape(params int[] dims)
    {
        _options.Shape = dims ?? Array.Empty<int>();
        return this;
    }

    public ChannelBuilder WithDataType(string dataType)
    {
        _options.DataType = dataType ?? "float32";
        return this;
    }

    public ChannelBuilder WithPreprocess(string pipelineName)
    {
        _options.Preprocess = pipelineName ?? "noop";
        return this;
    }

    public ChannelBuilder WithBatching(int maxBatch, int timeoutMs)
    {
        _options.MaxBatch = Math.Max(1, maxBatch);
        _options.BatchTimeoutMs = Math.Max(0, timeoutMs);
        return this;
    }

    public ChannelBuilder WithNormalization(float mean, float std)
    {
        _options.NormalizeMean = mean;
        _options.NormalizeStd = Math.Max(1e-6f, std);
        return this;
    }

    public ChannelBuilder WithScenePriority(SceneLevel level)
    {
        _options.ScenePriority = level;
        return this;
    }

    public ChannelBuilder WithWeight(int weight)
    {
        _options.Weight = Math.Max(1, weight);
        return this;
    }

    public ChannelBuilder WithMaxInFlight(int maxInFlight)
    {
        _options.MaxInFlight = Math.Max(1, maxInFlight);
        return this;
    }

    public ChannelBuilder WithQueueDepth(int maxQueueDepth)
    {
        _options.MaxQueueDepth = Math.Max(1, maxQueueDepth);
        return this;
    }

    public ChannelBuilder WithFpsRange(int minFps, int maxFps)
    {
        _options.MinFps = Math.Max(1, minFps);
        _options.MaxFps = Math.Max(_options.MinFps, maxFps);
        return this;
    }
}

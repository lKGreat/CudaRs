using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS;

public sealed class PipelineBuilder
{
    private readonly Dictionary<string, ChannelOptions> _channels = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _preprocessStages = new();
    private readonly List<string> _inferStages = new();
    private readonly List<string> _postprocessStages = new();
    private readonly Dictionary<string, ModelOptions> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly ExecutionOptions _execution = new();
    private readonly GpuMemoryConfig _memoryConfig = new();
    private ResultRoutingMode _routingMode = ResultRoutingMode.PerChannel;
    private string _name = "DefaultPipeline";

    public PipelineBuilder WithName(string name)
    {
        _name = string.IsNullOrWhiteSpace(name) ? "DefaultPipeline" : name.Trim();
        return this;
    }

    public PipelineBuilder WithChannel(string name, Action<ChannelBuilder> configure)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Channel name is required.", nameof(name));

        if (!_channels.TryGetValue(name, out var options))
        {
            options = new ChannelOptions { Name = name };
            _channels.Add(name, options);
        }

        configure?.Invoke(new ChannelBuilder(options));
        return this;
    }

    public PipelineBuilder WithModel(string modelId, Action<ModelBuilder>? configure = null)
    {
        var id = string.IsNullOrWhiteSpace(modelId) ? "default" : modelId.Trim();
        if (!_models.TryGetValue(id, out var model))
        {
            model = new ModelOptions { ModelId = id };
            _models.Add(id, model);
        }

        configure?.Invoke(new ModelBuilder(model));
        return this;
    }

    public PipelineBuilder WithPreprocessStage(string name)
    {
        if (!string.IsNullOrWhiteSpace(name))
            _preprocessStages.Add(name.Trim());
        return this;
    }

    public PipelineBuilder WithInferStage(string name)
    {
        if (!string.IsNullOrWhiteSpace(name))
            _inferStages.Add(name.Trim());
        return this;
    }

    public PipelineBuilder WithPostprocessStage(string name)
    {
        if (!string.IsNullOrWhiteSpace(name))
            _postprocessStages.Add(name.Trim());
        return this;
    }

    public PipelineBuilder WithResultRouting(ResultRoutingMode mode)
    {
        _routingMode = mode;
        return this;
    }

    public PipelineBuilder WithExecution(Action<ExecutionOptions> configure)
    {
        configure?.Invoke(_execution);
        return this;
    }

    public PipelineBuilder WithGpuMemory(Action<GpuMemoryConfig> configure)
    {
        configure?.Invoke(_memoryConfig);
        return this;
    }

    public PipelineRun Build()
    {
        if (_channels.Count == 0)
            throw new InvalidOperationException("At least one channel must be configured.");

        foreach (var channel in _channels.Values)
        {
            if (channel.MinFps <= 0)
                throw new InvalidOperationException($"Channel '{channel.Name}' MinFps must be > 0.");
            if (channel.MaxFps < channel.MinFps)
                throw new InvalidOperationException($"Channel '{channel.Name}' MaxFps must be >= MinFps.");
        }

        var modelAssignments = AssignModelDevices();
        var definition = new PipelineDefinition
        {
            Name = _name,
            Channels = new Dictionary<string, ChannelOptions>(_channels, StringComparer.OrdinalIgnoreCase),
            Models = modelAssignments,
            MemoryConfig = _memoryConfig,
            PreprocessStages = _preprocessStages.ToArray(),
            InferStages = _inferStages.ToArray(),
            PostprocessStages = _postprocessStages.ToArray(),
            RoutingMode = _routingMode,
            Execution = _execution,
        };

        return new PipelineRun(definition);
    }

    private IReadOnlyDictionary<string, ModelOptions> AssignModelDevices()
    {
        if (_models.Count == 0)
            return new Dictionary<string, ModelOptions>();

        var deviceIds = _memoryConfig.DeviceIds.Length == 0
            ? new[] { 0 }
            : _memoryConfig.DeviceIds;

        if (_memoryConfig.DeviceSelection == GpuDeviceSelection.Fixed)
        {
            foreach (var model in _models.Values)
            {
                if (!model.DeviceId.HasValue)
                    model.DeviceId = deviceIds[0];
            }

            return new Dictionary<string, ModelOptions>(_models, StringComparer.OrdinalIgnoreCase);
        }

        var index = 0;
        foreach (var model in _models.Values)
        {
            if (model.DeviceId.HasValue)
                continue;

            if (_memoryConfig.DeviceSelection == GpuDeviceSelection.RoundRobin)
            {
                model.DeviceId = deviceIds[index % deviceIds.Length];
                index++;
            }
            else
            {
                model.DeviceId = deviceIds[0];
            }
        }

        return new Dictionary<string, ModelOptions>(_models, StringComparer.OrdinalIgnoreCase);
    }
}

using System;
using System.Collections.Generic;

namespace CudaRS;

public sealed class PipelineBuilder
{
    private readonly Dictionary<string, ChannelOptions> _channels = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, ModelOptions> _models = new(StringComparer.OrdinalIgnoreCase);
    private readonly ExecutionOptions _execution = new();
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

    public PipelineBuilder WithExecution(Action<ExecutionOptions> configure)
    {
        configure?.Invoke(_execution);
        return this;
    }

    public PipelineRun Build()
    {
        if (_channels.Count == 0)
            throw new InvalidOperationException("At least one channel must be configured.");
        if (_models.Count == 0)
            throw new InvalidOperationException("At least one model must be configured.");

        var definition = new PipelineDefinition
        {
            Name = _name,
            Channels = new Dictionary<string, ChannelOptions>(_channels, StringComparer.OrdinalIgnoreCase),
            Models = new Dictionary<string, ModelOptions>(_models, StringComparer.OrdinalIgnoreCase),
            Execution = _execution,
        };

        return new PipelineRun(definition);
    }
}

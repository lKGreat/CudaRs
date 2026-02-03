using System;

namespace CudaRS;

public sealed class ModelBuilder
{
    private readonly ModelOptions _options;

    internal ModelBuilder(ModelOptions options)
    {
        _options = options;
    }

    public ModelBuilder WithKind(ModelKind kind)
    {
        _options.Kind = kind;
        return this;
    }

    public ModelBuilder WithConfigJson(string json)
    {
        _options.ConfigJson = string.IsNullOrWhiteSpace(json) ? "{}" : json;
        return this;
    }

    public ModelBuilder WithPipeline(string pipelineId, PipelineKind kind, string configJson)
    {
        var id = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId.Trim();
        _options.Pipelines.Add(new PipelineOptions
        {
            PipelineId = id,
            Kind = kind,
            ConfigJson = string.IsNullOrWhiteSpace(configJson) ? "{}" : configJson,
        });
        return this;
    }
}

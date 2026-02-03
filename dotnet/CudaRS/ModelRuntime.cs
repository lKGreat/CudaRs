using System;
using System.Collections.Generic;
using CudaRS.Interop;

namespace CudaRS;

internal sealed class ModelRuntime : IDisposable
{
    public ModelRuntime(ModelOptions options, ModelHandle handle, Dictionary<string, PipelineHandle> pipelines)
    {
        Options = options;
        Handle = handle;
        Pipelines = pipelines;
    }

    public ModelOptions Options { get; }
    public ModelHandle Handle { get; }
    public Dictionary<string, PipelineHandle> Pipelines { get; }

    public PipelineHandle GetPipeline(string pipelineId)
    {
        var key = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId.Trim();
        if (Pipelines.TryGetValue(key, out var pipeline))
            return pipeline;
        throw new KeyNotFoundException($"Pipeline '{key}' not found for model '{Options.ModelId}'.");
    }

    public void Dispose()
    {
        foreach (var pipeline in Pipelines.Values)
            pipeline.Dispose();
        Pipelines.Clear();
    }
}

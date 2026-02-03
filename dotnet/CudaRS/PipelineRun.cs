using System;
using System.Collections.Generic;
using System.Diagnostics;
using CudaRS.Interop;

namespace CudaRS;

public sealed class PipelineRun : IDisposable
{
    private readonly PipelineDefinition _definition;
    private readonly ModelHub _modelHub;
    private readonly Dictionary<string, ModelRuntime> _models = new(StringComparer.OrdinalIgnoreCase);

    internal PipelineRun(PipelineDefinition definition)
    {
        _definition = definition ?? throw new ArgumentNullException(nameof(definition));
        _modelHub = new ModelHub();
        InitializeModels();
    }

    public PipelineDefinition Definition => _definition;

    internal IReadOnlyDictionary<string, ModelRuntime> Models => _models;

    public PipelineHandle GetPipeline(string modelId, string pipelineId)
    {
        if (!_models.TryGetValue(modelId, out var runtime))
            throw new KeyNotFoundException($"Model '{modelId}' not found.");
        return runtime.GetPipeline(pipelineId);
    }

    public RunResult Run(PipelineInput input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var stopwatch = Stopwatch.StartNew();
        var diagnostics = new List<string>();
        var outputs = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        var modelOutputs = new Dictionary<string, IReadOnlyDictionary<string, object>>(StringComparer.OrdinalIgnoreCase);

        foreach (var channel in _definition.Channels.Keys)
        {
            if (!input.Channels.TryGetValue(channel, out var payload))
            {
                diagnostics.Add($"Missing input for channel '{channel}'.");
                continue;
            }

            outputs[channel] = payload.Payload;
        }

        foreach (var model in _definition.Models.Values)
        {
            modelOutputs[model.ModelId] = new Dictionary<string, object>(outputs, StringComparer.OrdinalIgnoreCase);
        }

        stopwatch.Stop();

        return new RunResult
        {
            PipelineName = _definition.Name,
            Success = diagnostics.Count == 0,
            Elapsed = stopwatch.Elapsed,
            PerChannelOutputs = outputs,
            ModelOutputs = modelOutputs,
            Diagnostics = diagnostics,
        };
    }

    public void Dispose()
    {
        foreach (var runtime in _models.Values)
            runtime.Dispose();
        _models.Clear();
        _modelHub.Dispose();
    }

    private void InitializeModels()
    {
        foreach (var model in _definition.Models.Values)
        {
            var handle = _modelHub.LoadModel(model);
            if (model.Pipelines.Count == 0)
            {
                throw new InvalidOperationException($"Model '{model.ModelId}' must define at least one pipeline.");
            }

            var pipelines = new Dictionary<string, PipelineHandle>(StringComparer.OrdinalIgnoreCase);
            foreach (var pipeline in model.Pipelines)
            {
                var handlePipeline = _modelHub.CreatePipeline(handle, pipeline);
                pipelines[pipeline.PipelineId] = handlePipeline;
            }

            _models[model.ModelId] = new ModelRuntime(model, handle, pipelines);
        }
    }
}

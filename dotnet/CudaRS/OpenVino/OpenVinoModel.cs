using System;
using CudaRS;
using CudaRS.Interop;

namespace CudaRS.OpenVino;

public sealed class OpenVinoModel : IDisposable
{
    private readonly ModelHub _hub;
    private readonly bool _ownsHub;
    private readonly ModelHandle _model;

    public OpenVinoModel(string modelId, OpenVinoModelConfig config, ModelHub? hub = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Model id is required.", nameof(modelId));
        if (config == null)
            throw new ArgumentNullException(nameof(config));
        if (string.IsNullOrWhiteSpace(config.ModelPath))
            throw new ArgumentException("ModelPath is required.", nameof(config));

        _hub = hub ?? new ModelHub();
        _ownsHub = hub == null;
        _model = _hub.LoadModel(new ModelOptions
        {
            ModelId = modelId,
            Kind = ModelKind.OpenVino,
            ConfigJson = config.ToJson()
        });
    }

    public OpenVinoPipeline CreatePipeline(string pipelineId, OpenVinoPipelineConfig? config = null)
    {
        var pipelineConfig = config ?? new OpenVinoPipelineConfig();
        var handle = _hub.CreatePipeline(_model, new PipelineOptions
        {
            PipelineId = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId,
            Kind = PipelineKind.OpenVinoTensor,
            ConfigJson = pipelineConfig.ToJson()
        });

        return new OpenVinoPipeline(handle);
    }

    public void Dispose()
    {
        if (_ownsHub)
            _hub.Dispose();
    }
}

using System;
using CudaRS.Interop;

namespace CudaRS.Ocr;

public sealed class OcrModel : IDisposable
{
    private readonly ModelHub _hub;
    private readonly bool _ownsHub;
    private readonly ModelHandle _model;

    public OcrModel(string modelId, OcrModelConfig config, ModelHub? hub = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Model id is required.", nameof(modelId));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        _hub = hub ?? new ModelHub();
        _ownsHub = hub == null;
        _model = _hub.LoadModel(new ModelOptions
        {
            ModelId = modelId,
            Kind = ModelKind.PaddleOcr,
            ConfigJson = config.ToJson()
        });
    }

    public OcrPipeline CreatePipeline(string pipelineId, OcrPipelineConfig? config = null)
    {
        var pipelineConfig = config ?? new OcrPipelineConfig();
        var handle = _hub.CreatePipeline(_model, new PipelineOptions
        {
            PipelineId = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId,
            Kind = PipelineKind.PaddleOcr,
            ConfigJson = pipelineConfig.ToJson()
        });

        return new OcrPipeline(handle);
    }

    public void Dispose()
    {
        if (_ownsHub)
            _hub.Dispose();
    }
}

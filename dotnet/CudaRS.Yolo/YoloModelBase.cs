using System;
using System.Collections.Generic;
using CudaRS;
using CudaRS.Interop;

namespace CudaRS.Yolo;

public abstract class YoloModelBase : IDisposable
{
    private readonly ModelHub _hub;
    private readonly bool _ownsHub;
    private readonly ModelHandle _modelHandle;
    private readonly Dictionary<string, YoloPipeline> _pipelines = new(StringComparer.OrdinalIgnoreCase);

    protected YoloModelBase(YoloModelDefinition definition, ModelHub? hub = null)
    {
        if (definition == null)
            throw new ArgumentNullException(nameof(definition));
        if (string.IsNullOrWhiteSpace(definition.ModelId))
            throw new ArgumentException("ModelId is required", nameof(definition));
        if (string.IsNullOrWhiteSpace(definition.ModelPath))
            throw new ArgumentException("ModelPath is required", nameof(definition));

        Definition = definition;
        _hub = hub ?? new ModelHub();
        _ownsHub = hub == null;

        var modelOptions = new ModelOptions
        {
            ModelId = definition.ModelId,
            Kind = ModelKind.Yolo,
            ConfigJson = YoloModelConfigJson.Build(definition),
        };

        _modelHandle = _hub.LoadModel(modelOptions);
        YoloModelRegistry.Register(definition);
    }

    public YoloModelDefinition Definition { get; }

    public string ModelId => Definition.ModelId;

    public YoloConfig Config => Definition.Config;

    public YoloPipeline CreatePipeline(string pipelineId = "default", YoloPipelineOptions? options = null, bool ownsHandle = true)
    {
        var id = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId.Trim();
        var cfg = options ?? new YoloPipelineOptions();
        var pipelineKind = cfg.Device == InferenceDevice.Cpu
            ? PipelineKind.YoloCpu
            : PipelineKind.YoloGpuThroughput;

        var pipelineOptions = new PipelineOptions
        {
            PipelineId = id,
            Kind = pipelineKind,
            ConfigJson = cfg.ToJson(),
        };

        var handle = _hub.CreatePipeline(_modelHandle, pipelineOptions);
        var pipeline = new YoloPipeline(handle, Config, ModelId, ownsHandle);
        _pipelines[id] = pipeline;
        return pipeline;
    }

    public YoloPipeline GetOrCreatePipeline(string pipelineId = "default")
    {
        var id = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId.Trim();
        if (_pipelines.TryGetValue(id, out var pipeline))
            return pipeline;
        return CreatePipeline(id, null, true);
    }

    public ModelInferenceResult Run(ReadOnlyMemory<byte> imageBytes, string channelId, long frameIndex = 0)
    {
        var pipeline = GetOrCreatePipeline();
        return pipeline.Run(imageBytes, channelId, frameIndex);
    }

    public void Dispose()
    {
        foreach (var pipeline in _pipelines.Values)
            pipeline.Dispose();
        _pipelines.Clear();

        if (_ownsHub)
            _hub.Dispose();
    }

    protected static YoloModelDefinition BuildDefinition(
        string modelId,
        string modelPath,
        YoloConfig? config,
        int deviceId,
        YoloVersion version)
    {
        var cfg = config ?? new YoloConfig();
        cfg.Version = version;
        YoloVersionAdapter.ApplyVersionDefaults(cfg);
        return new YoloModelDefinition
        {
            ModelId = modelId,
            ModelPath = modelPath,
            Config = cfg,
            DeviceId = deviceId,
        };
    }
}

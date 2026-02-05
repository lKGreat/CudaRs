using System;
using System.Collections.Generic;
using CudaRS.Core;
using CudaRS.Interop;
using CudaRS.Native;

namespace CudaRS;

public sealed class ModelHub : IDisposable
{
    private readonly ModelManagerHandle _manager;
    private readonly Dictionary<string, ModelHandle> _models = new(StringComparer.OrdinalIgnoreCase);

    public ModelHub()
    {
        _manager = new ModelManagerHandle();
    }

    public ModelHandle LoadModel(ModelOptions options)
    {
        if (options == null)
            throw new ArgumentNullException(nameof(options));
        if (options.Kind == ModelKind.Unknown)
            throw new ArgumentException("Model kind is required.", nameof(options));

        if (_models.TryGetValue(options.ModelId, out var existing))
            return existing;

        using var id = new Utf8Buffer(options.ModelId);
        using var config = new Utf8Buffer(options.ConfigJson);

        var spec = new SdkModelSpec
        {
            IdPtr = id.Pointer,
            IdLen = id.Length,
            Kind = MapKind(options.Kind),
            ConfigJsonPtr = config.Pointer,
            ConfigJsonLen = config.Length,
        };

        SdkCheck.ThrowIfError(SdkNative.ModelManagerLoadModel(_manager.Value, in spec, out var handle));
        var modelHandle = new ModelHandle(handle);
        _models[options.ModelId] = modelHandle;
        return modelHandle;
    }

    public PipelineHandle CreatePipeline(ModelHandle modelHandle, PipelineOptions options)
    {
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        using var id = new Utf8Buffer(options.PipelineId);
        using var config = new Utf8Buffer(options.ConfigJson);

        var spec = new SdkPipelineSpec
        {
            IdPtr = id.Pointer,
            IdLen = id.Length,
            Kind = MapPipelineKind(options.Kind),
            ConfigJsonPtr = config.Pointer,
            ConfigJsonLen = config.Length,
        };

        SdkCheck.ThrowIfError(SdkNative.ModelCreatePipeline(modelHandle.Value, in spec, out var pipelineHandle));
        return new PipelineHandle(pipelineHandle);
    }

    private static SdkModelKind MapKind(ModelKind kind)
    {
        return kind switch
        {
            ModelKind.Yolo => SdkModelKind.Yolo,
            ModelKind.PaddleOcr => SdkModelKind.PaddleOcr,
            ModelKind.OpenVino => SdkModelKind.OpenVino,
            ModelKind.OpenVinoOcr => SdkModelKind.OpenVinoOcr,
            _ => SdkModelKind.Unknown,
        };
    }

    private static SdkPipelineKind MapPipelineKind(PipelineKind kind)
    {
        return kind switch
        {
            PipelineKind.YoloCpu => SdkPipelineKind.YoloCpu,
            PipelineKind.YoloGpuThroughput => SdkPipelineKind.YoloGpuThroughput,
            PipelineKind.PaddleOcr => SdkPipelineKind.PaddleOcr,
            PipelineKind.YoloOpenVino => SdkPipelineKind.YoloOpenVino,
            PipelineKind.OpenVinoTensor => SdkPipelineKind.OpenVinoTensor,
            PipelineKind.OpenVinoOcr => SdkPipelineKind.OpenVinoOcr,
            _ => SdkPipelineKind.Unknown,
        };
    }

    public void Dispose()
    {
        _manager.Dispose();
    }
}

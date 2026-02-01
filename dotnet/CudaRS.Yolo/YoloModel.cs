using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

/// <summary>
/// YOLO model wrapper that handles preprocessing, inference, and postprocessing.
/// </summary>
public sealed class YoloModel : IDisposable
{
    private readonly IInferenceBackend _backend;
    private bool _disposed;

    public string ModelId { get; }
    public string ModelPath { get; }
    public YoloConfig Config { get; }
    public int DeviceId { get; }

    private YoloModel(string modelId, string modelPath, YoloConfig config, IInferenceBackend backend)
    {
        ModelId = modelId;
        ModelPath = modelPath;
        Config = config;
        DeviceId = backend.DeviceId;
        _backend = backend;
    }

    /// <summary>
    /// Creates a YOLO model from the given definition.
    /// </summary>
    public static YoloModel Create(YoloModelDefinition definition)
    {
        if (string.IsNullOrEmpty(definition.ModelId))
            throw new ArgumentException("ModelId is required");
        if (string.IsNullOrEmpty(definition.ModelPath))
            throw new ArgumentException("ModelPath is required");

        // Apply version-specific defaults
        YoloVersionAdapter.ApplyVersionDefaults(definition.Config);

        // Create backend
        var backend = InferenceBackendFactory.Create(
            definition.ModelPath,
            definition.Config.Backend,
            definition.DeviceId);

        return new YoloModel(definition.ModelId, definition.ModelPath, definition.Config, backend);
    }

    /// <summary>
    /// Creates a YOLO model with the specified backend.
    /// </summary>
    public static YoloModel Create(
        string modelId,
        string modelPath,
        YoloConfig config,
        int deviceId = 0)
    {
        return Create(new YoloModelDefinition
        {
            ModelId = modelId,
            ModelPath = modelPath,
            Config = config,
            DeviceId = deviceId,
        });
    }

    /// <summary>
    /// Runs inference on the given image.
    /// </summary>
    public ModelInferenceResult Run(string channelId, YoloImage image, long frameIndex = 0)
    {
        // Preprocess
        var preprocess = YoloPreprocessor.Letterbox(image, Config.InputWidth, Config.InputHeight);

        // Run inference
        var backendResult = _backend.Run(preprocess.Input, preprocess.InputShape);

        // Postprocess
        return YoloPostprocessor.Decode(ModelId, Config, backendResult, preprocess, channelId, frameIndex);
    }

    /// <summary>
    /// Runs inference on raw preprocessed input.
    /// </summary>
    public BackendResult RunRaw(ReadOnlySpan<float> input, int[] shape)
        => _backend.Run(input, shape);

    public void Dispose()
    {
        if (!_disposed)
        {
            _backend.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Definition for creating a YOLO model.
/// </summary>
public sealed class YoloModelDefinition
{
    public string ModelId { get; init; } = string.Empty;
    public string ModelPath { get; init; } = string.Empty;
    public YoloConfig Config { get; init; } = new YoloConfig();
    public int DeviceId { get; init; }
}


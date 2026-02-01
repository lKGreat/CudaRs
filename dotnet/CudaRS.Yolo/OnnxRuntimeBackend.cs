using System;
using System.Collections.Generic;
using CudaRS.Core;

namespace CudaRS.Yolo;

/// <summary>
/// ONNX Runtime inference backend for YOLO models.
/// </summary>
public sealed class OnnxRuntimeBackend : IInferenceBackend
{
    private readonly OnnxRuntimeSession _session;
    private readonly string _modelPath;
    private readonly int _deviceId;
    private int[]? _inputShape;
    private bool _disposed;

    private OnnxRuntimeBackend(OnnxRuntimeSession session, string modelPath, int deviceId)
    {
        _session = session;
        _modelPath = modelPath;
        _deviceId = deviceId;
    }

    /// <summary>
    /// Loads an ONNX model from file.
    /// </summary>
    public static OnnxRuntimeBackend Load(string modelPath, int deviceId = 0)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentNullException(nameof(modelPath));

        var session = new OnnxRuntimeSession(modelPath, deviceId);
        return new OnnxRuntimeBackend(session, modelPath, deviceId);
    }

    public int[] InputShape
    {
        get => _inputShape ??= new[] { 1, 3, 640, 640 }; // Default YOLO input
        set => _inputShape = value;
    }

    public int DeviceId => _deviceId;
    public string ModelPath => _modelPath;

    public BackendResult Run(ReadOnlySpan<float> input, int[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentNullException(nameof(shape));

        var inputArray = input.ToArray();
        var tensors = _session.Run(inputArray, shape);
        var outputs = new List<TensorOutput>(tensors.Count);

        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            outputs.Add(new TensorOutput
            {
                Name = $"output_{i}",
                Data = t.Data,
                Shape = t.Shape,
            });
        }

        return new BackendResult { Outputs = outputs };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _session.Dispose();
            _disposed = true;
        }
    }
}

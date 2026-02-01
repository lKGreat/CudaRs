using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

/// <summary>
/// Output tensor from inference backend.
/// </summary>
public sealed class TensorOutput
{
    public string Name { get; init; } = string.Empty;
    public float[] Data { get; init; } = Array.Empty<float>();
    public int[] Shape { get; init; } = Array.Empty<int>();
}

/// <summary>
/// Result from backend inference containing output tensors.
/// </summary>
public sealed class BackendResult
{
    public IReadOnlyList<TensorOutput> Outputs { get; init; } = Array.Empty<TensorOutput>();
}

/// <summary>
/// Interface for YOLO inference backends.
/// </summary>
public interface IInferenceBackend : IDisposable
{
    /// <summary>Input tensor shape [batch, channels, height, width].</summary>
    int[] InputShape { get; }

    /// <summary>CUDA device ID (or -1 for CPU/non-CUDA backends).</summary>
    int DeviceId { get; }

    /// <summary>Runs inference with the given input tensor.</summary>
    BackendResult Run(ReadOnlySpan<float> input, int[] shape);
}

/// <summary>
/// Factory for creating inference backends.
/// </summary>
public static class InferenceBackendFactory
{
    /// <summary>
    /// Creates an inference backend based on model file and configuration.
    /// </summary>
    public static IInferenceBackend Create(
        string modelPath,
        InferenceBackend backend = InferenceBackend.Auto,
        int deviceId = 0)
    {
        var actualBackend = backend == InferenceBackend.Auto
            ? DetectBackend(modelPath)
            : backend;

        return actualBackend switch
        {
            InferenceBackend.OnnxRuntime => OnnxRuntimeBackend.Load(modelPath, deviceId),
            InferenceBackend.TensorRT => CreateTensorRtBackend(modelPath, deviceId),
            InferenceBackend.TorchScript => TorchScriptBackend.Load(modelPath, deviceId),
            InferenceBackend.OpenVino => OpenVinoBackend.Load(modelPath, new OpenVinoOptions
            {
                Device = deviceId >= 0 ? CudaRS.Native.CudaRsOvDevice.Gpu : CudaRS.Native.CudaRsOvDevice.Cpu,
                DeviceIndex = Math.Max(0, deviceId),
            }),
            _ => throw new NotSupportedException($"Backend {actualBackend} is not supported"),
        };
    }

    private static InferenceBackend DetectBackend(string modelPath)
    {
        var ext = System.IO.Path.GetExtension(modelPath).ToLowerInvariant();
        return ext switch
        {
            ".onnx" => InferenceBackend.OnnxRuntime,
            ".engine" or ".trt" or ".plan" => InferenceBackend.TensorRT,
            ".pt" or ".pth" or ".torchscript" => InferenceBackend.TorchScript,
            ".xml" or ".bin" => InferenceBackend.OpenVino,
            _ => InferenceBackend.OnnxRuntime, // Default
        };
    }

    private static IInferenceBackend CreateTensorRtBackend(string modelPath, int deviceId)
    {
        var ext = System.IO.Path.GetExtension(modelPath).ToLowerInvariant();

        // If it's an ONNX file, build TensorRT engine
        if (ext == ".onnx")
            return TensorRtBackend.BuildFromOnnx(modelPath, deviceId);

        // Otherwise load serialized engine
        return TensorRtBackend.LoadEngine(modelPath, deviceId);
    }
}

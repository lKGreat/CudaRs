using System;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Yolo;

/// <summary>
/// TensorRT inference backend for YOLO models.
/// Supports loading both ONNX (builds engine) and serialized TRT engines.
/// </summary>
public sealed class TensorRtBackend : IInferenceBackend
{
    private ulong _handle;
    private bool _disposed;
    private readonly int _deviceId;
    private readonly int[] _inputShape;

    private TensorRtBackend(ulong handle, int deviceId, int[] inputShape)
    {
        _handle = handle;
        _deviceId = deviceId;
        _inputShape = inputShape;
    }

    /// <summary>
    /// Builds a TensorRT engine from an ONNX model file.
    /// </summary>
    public static TensorRtBackend BuildFromOnnx(
        string onnxPath,
        int deviceId = 0,
        TensorRtBuildOptions? options = null)
    {
        var opts = options ?? new TensorRtBuildOptions();

        var config = new CudaRsTrtBuildConfig
        {
            Fp16Enabled = opts.EnableFp16 ? 1 : 0,
            Int8Enabled = opts.EnableInt8 ? 1 : 0,
            MaxBatchSize = opts.MaxBatchSize,
            WorkspaceSizeMb = opts.WorkspaceSizeMb,
            DlaCore = opts.DlaCore,
        };

        var result = CudaRsNative.TrtBuildEngine(onnxPath, deviceId, in config, out var handle);
        if (result != CudaRsResult.Success)
            throw new CudaException($"Failed to build TensorRT engine: {result}");

        var inputShape = GetInputShape(handle);
        return new TensorRtBackend(handle, deviceId, inputShape);
    }

    /// <summary>
    /// Loads a serialized TensorRT engine from file.
    /// </summary>
    public static TensorRtBackend LoadEngine(string enginePath, int deviceId = 0)
    {
        var result = CudaRsNative.TrtLoadEngine(enginePath, deviceId, out var handle);
        if (result != CudaRsResult.Success)
            throw new CudaException($"Failed to load TensorRT engine: {result}");

        var inputShape = GetInputShape(handle);
        return new TensorRtBackend(handle, deviceId, inputShape);
    }

    /// <summary>
    /// Saves the TensorRT engine to file for faster loading next time.
    /// </summary>
    public void SaveEngine(string path)
    {
        var result = CudaRsNative.TrtSaveEngine(_handle, path);
        if (result != CudaRsResult.Success)
            throw new CudaException($"Failed to save TensorRT engine: {result}");
    }

    public BackendResult Run(ReadOnlySpan<float> input, int[] shape)
    {
        unsafe
        {
            fixed (float* inputPtr = input)
            {
                var result = CudaRsNative.TrtRun(
                    _handle,
                    inputPtr,
                    (ulong)input.Length,
                    out var tensorsPtr,
                    out var tensorCount);

                if (result != CudaRsResult.Success)
                    throw new CudaException($"TensorRT inference failed: {result}");

                return ExtractTensors(tensorsPtr, tensorCount);
            }
        }
    }

    public int[] InputShape => _inputShape;
    public int DeviceId => _deviceId;

    private static int[] GetInputShape(ulong handle)
    {
        unsafe
        {
            Span<long> shape = stackalloc long[8];
            fixed (long* shapePtr = shape)
            {
                var result = CudaRsNative.TrtGetInputInfo(handle, 0, shapePtr, out var shapeLen, 8);
                if (result != CudaRsResult.Success)
                    return new[] { 1, 3, 640, 640 }; // Default

                var arr = new int[shapeLen];
                for (int i = 0; i < shapeLen; i++)
                    arr[i] = (int)shape[i];
                return arr;
            }
        }
    }

    private static BackendResult ExtractTensors(IntPtr tensorsPtr, ulong tensorCount)
    {
        var outputs = new System.Collections.Generic.List<TensorOutput>();

        unsafe
        {
            var tensors = (CudaRsTrtTensor*)tensorsPtr;
            for (ulong i = 0; i < tensorCount; i++)
            {
                var t = tensors[i];

                var dataLen = (int)t.DataLen;
                var shapeLen = (int)t.ShapeLen;

                var data = new float[dataLen];
                var shape = new int[shapeLen];

                Marshal.Copy(t.Data, data, 0, dataLen);

                var shapeSpan = new Span<long>((void*)t.Shape, shapeLen);
                for (int j = 0; j < shapeLen; j++)
                    shape[j] = (int)shapeSpan[j];

                outputs.Add(new TensorOutput { Data = data, Shape = shape });
            }
        }

        CudaRsNative.TrtFreeTensors(tensorsPtr, tensorCount);

        return new BackendResult { Outputs = outputs };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            CudaRsNative.TrtDestroy(_handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// TensorRT engine build options.
/// </summary>
public sealed class TensorRtBuildOptions
{
    public bool EnableFp16 { get; set; } = true;
    public bool EnableInt8 { get; set; } = false;
    public int MaxBatchSize { get; set; } = 1;
    public int WorkspaceSizeMb { get; set; } = 1024;
    public int DlaCore { get; set; } = -1; // -1 = disabled
}

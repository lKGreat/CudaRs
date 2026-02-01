using System;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Yolo;

/// <summary>
/// TorchScript/PyTorch inference backend for YOLO models.
/// Supports .pt and .torchscript model files.
/// </summary>
public sealed class TorchScriptBackend : IInferenceBackend
{
    private ulong _handle;
    private bool _disposed;
    private readonly int _deviceId;
    private readonly int[] _inputShape;

    private TorchScriptBackend(ulong handle, int deviceId, int[] inputShape)
    {
        _handle = handle;
        _deviceId = deviceId;
        _inputShape = inputShape;
    }

    /// <summary>
    /// Checks if CUDA is available for PyTorch.
    /// </summary>
    public static bool IsCudaAvailable() => CudaRsNative.TorchCudaAvailable() != 0;

    /// <summary>
    /// Gets the number of CUDA devices available to PyTorch.
    /// </summary>
    public static int CudaDeviceCount() => CudaRsNative.TorchCudaDeviceCount();

    /// <summary>
    /// Loads a TorchScript model from file.
    /// </summary>
    /// <param name="modelPath">Path to .pt or .torchscript file</param>
    /// <param name="deviceId">CUDA device ID (-1 for CPU)</param>
    /// <param name="inputShape">Expected input shape (optional, defaults to [1,3,640,640])</param>
    public static TorchScriptBackend Load(string modelPath, int deviceId = 0, int[]? inputShape = null)
    {
        var result = CudaRsNative.TorchLoad(modelPath, deviceId, out var handle);
        if (result != CudaRsResult.Success)
            throw new CudaException($"Failed to load TorchScript model: {result}");

        // Set to eval mode
        var evalResult = CudaRsNative.TorchEval(handle);
        if (evalResult != CudaRsResult.Success)
            throw new CudaException($"Failed to set eval mode: {evalResult}");

        var shape = inputShape ?? new[] { 1, 3, 640, 640 };
        return new TorchScriptBackend(handle, deviceId, shape);
    }

    public BackendResult Run(ReadOnlySpan<float> input, int[] shape)
    {
        unsafe
        {
            fixed (float* inputPtr = input)
            fixed (long* shapePtr = ToLongArray(shape))
            {
                var result = CudaRsNative.TorchRun(
                    _handle,
                    inputPtr,
                    (ulong)input.Length,
                    shapePtr,
                    (ulong)shape.Length,
                    out var tensorsPtr,
                    out var tensorCount);

                if (result != CudaRsResult.Success)
                    throw new CudaException($"TorchScript inference failed: {result}");

                return ExtractTensors(tensorsPtr, tensorCount);
            }
        }
    }

    /// <summary>
    /// Runs inference with multiple input tensors.
    /// </summary>
    public BackendResult RunMulti(TensorInput[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input required");

        unsafe
        {
            // Pin all data arrays
            var handles = new GCHandle[inputs.Length * 2];
            var descs = new CudaRsTorchInputDesc[inputs.Length];

            try
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    handles[i * 2] = GCHandle.Alloc(inputs[i].Data, GCHandleType.Pinned);
                    var longShape = ToLongArray(inputs[i].Shape);
                    handles[i * 2 + 1] = GCHandle.Alloc(longShape, GCHandleType.Pinned);

                    descs[i] = new CudaRsTorchInputDesc
                    {
                        Data = handles[i * 2].AddrOfPinnedObject(),
                        DataLen = (ulong)inputs[i].Data.Length,
                        Shape = handles[i * 2 + 1].AddrOfPinnedObject(),
                        ShapeLen = (ulong)inputs[i].Shape.Length,
                    };
                }

                fixed (CudaRsTorchInputDesc* descsPtr = descs)
                {
                    var result = CudaRsNative.TorchRunMulti(
                        _handle,
                        descsPtr,
                        inputs.Length,
                        out var tensorsPtr,
                        out var tensorCount);

                    if (result != CudaRsResult.Success)
                        throw new CudaException($"TorchScript multi-input inference failed: {result}");

                    return ExtractTensors(tensorsPtr, tensorCount);
                }
            }
            finally
            {
                foreach (var h in handles)
                {
                    if (h.IsAllocated)
                        h.Free();
                }
            }
        }
    }

    public int[] InputShape => _inputShape;
    public int DeviceId => _deviceId;

    private static long[] ToLongArray(int[] arr)
    {
        var result = new long[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = arr[i];
        return result;
    }

    private static BackendResult ExtractTensors(IntPtr tensorsPtr, ulong tensorCount)
    {
        var outputs = new System.Collections.Generic.List<TensorOutput>();

        unsafe
        {
            var tensors = (CudaRsTorchTensor*)tensorsPtr;
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

        CudaRsNative.TorchFreeTensors(tensorsPtr, tensorCount);

        return new BackendResult { Outputs = outputs };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            CudaRsNative.TorchDestroy(_handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// Input tensor for multi-input inference.
/// </summary>
public sealed class TensorInput
{
    public required float[] Data { get; init; }
    public required int[] Shape { get; init; }
}

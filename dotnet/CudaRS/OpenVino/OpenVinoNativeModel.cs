using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using CudaRS.Native;

namespace CudaRS.OpenVino;

public sealed class OpenVinoNativeConfig
{
    public string Device { get; set; } = "cpu";
    public int NumStreams { get; set; }
    public bool EnableProfiling { get; set; }
    public string? PropertiesJson { get; set; }
}

public sealed class OpenVinoNativeModel : IDisposable
{
    private ulong _handle;
    private bool _disposed;

    public OpenVinoNativeModel(string modelPath, OpenVinoNativeConfig? config = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path is required.", nameof(modelPath));

        config ??= new OpenVinoNativeConfig();
        var (device, deviceIndex) = ParseDevice(config.Device);

        var native = new CudaRsOvConfig
        {
            Device = device,
            DeviceIndex = deviceIndex,
            NumStreams = config.NumStreams,
            EnableProfiling = config.EnableProfiling ? 1 : 0,
            PropertiesJsonPtr = IntPtr.Zero,
            PropertiesJsonLen = 0,
        };

        byte[]? jsonBytes = null;
        if (!string.IsNullOrWhiteSpace(config.PropertiesJson))
        {
            jsonBytes = Encoding.UTF8.GetBytes(config.PropertiesJson);
        }

        var pathBytes = Encoding.UTF8.GetBytes(modelPath + "\0");
        unsafe
        {
            fixed (byte* pathPtr = pathBytes)
            fixed (byte* jsonPtr = jsonBytes)
            {
                if (jsonBytes is { Length: > 0 })
                {
                    native.PropertiesJsonPtr = (IntPtr)jsonPtr;
                    native.PropertiesJsonLen = (nuint)jsonBytes.Length;
                }

                var res = SdkNative.OpenVinoLoad(pathPtr, in native, out _handle);
                ThrowIfError(res, "load");
            }
        }
    }

    public OpenVinoAsyncQueue CreateAsyncQueue() => new(this);

    internal ulong Handle => _handle;

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_handle != 0)
        {
            var res = SdkNative.OpenVinoDestroy(_handle);
            _handle = 0;
            if (res != CudaRsResult.Success)
                throw new InvalidOperationException($"OpenVINO destroy failed: {res}");
        }
    }

    private static (CudaRsOvDevice Device, int DeviceIndex) ParseDevice(string? device)
    {
        var d = (device ?? string.Empty).Trim().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(d) || d == "auto")
            return (CudaRsOvDevice.Auto, 0);
        if (d == "cpu")
            return (CudaRsOvDevice.Cpu, 0);
        if (d == "gpu")
            return (CudaRsOvDevice.Gpu, 0);
        if (d.StartsWith("gpu:") || d.StartsWith("gpu."))
        {
            var idx = int.Parse(d[4..]);
            return (CudaRsOvDevice.GpuIndex, idx);
        }
        if (d == "npu")
            return (CudaRsOvDevice.Npu, 0);

        throw new ArgumentException("Invalid OpenVINO device.", nameof(device));
    }

    private static void ThrowIfError(CudaRsResult res, string step)
    {
        if (res == CudaRsResult.Success)
            return;
        throw new InvalidOperationException($"OpenVINO {step} failed: {res}");
    }
}

public sealed class OpenVinoAsyncQueue
{
    private readonly OpenVinoNativeModel _model;

    internal OpenVinoAsyncQueue(OpenVinoNativeModel model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    public int Submit(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape)
    {
        if (input.IsEmpty)
            throw new ArgumentException("Input required.", nameof(input));
        if (shape.IsEmpty)
            throw new ArgumentException("Shape required.", nameof(shape));

        if (!MemoryMarshal.TryGetArray(input, out var inputSegment))
            inputSegment = new ArraySegment<float>(input.ToArray());
        if (!MemoryMarshal.TryGetArray(shape, out var shapeSegment))
            shapeSegment = new ArraySegment<long>(shape.ToArray());

        unsafe
        {
            fixed (float* inputPtr = inputSegment.Array)
            fixed (long* shapePtr = shapeSegment.Array)
            {
                var dataPtr = inputPtr + inputSegment.Offset;
                var shapePtrOffset = shapePtr + shapeSegment.Offset;
                var res = SdkNative.OpenVinoAsyncQueueSubmit(
                    _model.Handle,
                    dataPtr,
                    (ulong)inputSegment.Count,
                    shapePtrOffset,
                    (ulong)shapeSegment.Count,
                    out var requestId);
                if (res != CudaRsResult.Success)
                    throw new InvalidOperationException($"OpenVINO async submit failed: {res}");
                return requestId;
            }
        }
    }

    public OpenVinoTensorOutput[] Wait(int requestId)
    {
        unsafe
        {
            var res = SdkNative.OpenVinoAsyncQueueWait(_model.Handle, requestId, out var tensors, out var count);
            if (res != CudaRsResult.Success)
                throw new InvalidOperationException($"OpenVINO async wait failed: {res}");

            try
            {
                return ConvertOutputs(tensors, count);
            }
            finally
            {
                _ = SdkNative.OpenVinoFreeTensors(tensors, count);
            }
        }
    }

    private static unsafe OpenVinoTensorOutput[] ConvertOutputs(CudaRsOvTensor* tensors, ulong count)
    {
        var outputs = new List<OpenVinoTensorOutput>((int)count);
        for (ulong i = 0; i < count; i++)
        {
            var t = tensors[i];
            var shapeLen = checked((int)t.ShapeLen);
            var dataLen = checked((int)t.DataLen);

            var shape = new long[shapeLen];
            if (shapeLen > 0)
                Marshal.Copy(t.Shape, shape, 0, shapeLen);

            var data = new float[dataLen];
            if (dataLen > 0)
                Marshal.Copy(t.Data, data, 0, dataLen);

            var intShape = new int[shapeLen];
            for (int s = 0; s < shapeLen; s++)
                intShape[s] = (int)shape[s];

            outputs.Add(new OpenVinoTensorOutput
            {
                Data = data,
                Shape = intShape,
            });
        }

        return outputs.ToArray();
    }
}

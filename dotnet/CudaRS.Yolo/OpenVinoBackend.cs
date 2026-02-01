using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Yolo;

/// <summary>
/// OpenVINO inference backend for YOLO models.
/// Supports ONNX, OpenVINO IR (.xml/.bin), and other formats.
/// </summary>
public sealed class OpenVinoBackend : IInferenceBackend
{
    private ulong _handle;
    private bool _disposed;
    private readonly string _device;
    private readonly int[] _inputShape;

    private OpenVinoBackend(ulong handle, string device, int[] inputShape)
    {
        _handle = handle;
        _device = device;
        _inputShape = inputShape;
    }

    /// <summary>
    /// Gets available OpenVINO devices.
    /// </summary>
    public static string[] GetAvailableDevices()
    {
        unsafe
        {
            const int maxDevices = 16;
            var devicePtrs = stackalloc IntPtr[maxDevices];

            var result = CudaRsNative.OvGetDevices(devicePtrs, out var count, maxDevices);
            if (result != CudaRsResult.Success)
                return Array.Empty<string>();

            var devices = new string[count];
            for (int i = 0; i < count; i++)
            {
                devices[i] = Marshal.PtrToStringAnsi(devicePtrs[i]) ?? "";
                Marshal.FreeHGlobal(devicePtrs[i]);
            }

            return devices;
        }
    }

    /// <summary>
    /// Loads an OpenVINO model from file.
    /// </summary>
    /// <param name="modelPath">Path to ONNX or OpenVINO IR (.xml) file</param>
    /// <param name="options">Inference options</param>
    public static OpenVinoBackend Load(string modelPath, OpenVinoOptions? options = null)
    {
        var opts = options ?? new OpenVinoOptions();

        var config = new CudaRsOvConfig
        {
            Device = opts.Device,
            DeviceIndex = opts.DeviceIndex,
            NumStreams = opts.NumStreams,
            EnableProfiling = opts.EnableProfiling ? 1 : 0,
        };

        var result = CudaRsNative.OvLoad(modelPath, in config, out var handle);
        if (result != CudaRsResult.Success)
            throw new CudaException($"Failed to load OpenVINO model: {result}");

        var inputShape = GetInputShape(handle);
        var deviceName = GetDeviceName(opts);

        return new OpenVinoBackend(handle, deviceName, inputShape);
    }

    public BackendResult Run(ReadOnlySpan<float> input, int[] shape)
    {
        unsafe
        {
            fixed (float* inputPtr = input)
            fixed (long* shapePtr = ToLongArray(shape))
            {
                var result = CudaRsNative.OvRun(
                    _handle,
                    inputPtr,
                    (ulong)input.Length,
                    shapePtr,
                    (ulong)shape.Length,
                    out var tensorsPtr,
                    out var tensorCount);

                if (result != CudaRsResult.Success)
                    throw new CudaException($"OpenVINO inference failed: {result}");

                return ExtractTensors(tensorsPtr, tensorCount);
            }
        }
    }

    /// <summary>
    /// Starts asynchronous inference.
    /// </summary>
    public void RunAsync(ReadOnlySpan<float> input, int[] shape)
    {
        unsafe
        {
            fixed (float* inputPtr = input)
            fixed (long* shapePtr = ToLongArray(shape))
            {
                var result = CudaRsNative.OvRunAsync(
                    _handle,
                    inputPtr,
                    (ulong)input.Length,
                    shapePtr,
                    (ulong)shape.Length);

                if (result != CudaRsResult.Success)
                    throw new CudaException($"OpenVINO async inference failed: {result}");
            }
        }
    }

    /// <summary>
    /// Waits for asynchronous inference to complete and returns results.
    /// </summary>
    public BackendResult Wait()
    {
        var result = CudaRsNative.OvWait(_handle, out var tensorsPtr, out var tensorCount);
        if (result != CudaRsResult.Success)
            throw new CudaException($"OpenVINO wait failed: {result}");

        return ExtractTensors(tensorsPtr, tensorCount);
    }

    public int[] InputShape => _inputShape;
    public string Device => _device;
    public int DeviceId => 0; // OpenVINO uses device name

    private static int[] GetInputShape(ulong handle)
    {
        unsafe
        {
            Span<long> shape = stackalloc long[8];
            fixed (long* shapePtr = shape)
            {
                var result = CudaRsNative.OvGetInputInfo(handle, 0, shapePtr, out var shapeLen, 8);
                if (result != CudaRsResult.Success)
                    return new[] { 1, 3, 640, 640 };

                var arr = new int[shapeLen];
                for (int i = 0; i < shapeLen; i++)
                    arr[i] = (int)shape[i];
                return arr;
            }
        }
    }

    private static string GetDeviceName(OpenVinoOptions opts)
        => opts.Device switch
        {
            CudaRsOvDevice.Cpu => "CPU",
            CudaRsOvDevice.Gpu => "GPU",
            CudaRsOvDevice.GpuIndex => $"GPU.{opts.DeviceIndex}",
            CudaRsOvDevice.Npu => "NPU",
            CudaRsOvDevice.Auto => "AUTO",
            _ => "CPU",
        };

    private static long[] ToLongArray(int[] arr)
    {
        var result = new long[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = arr[i];
        return result;
    }

    private static BackendResult ExtractTensors(IntPtr tensorsPtr, ulong tensorCount)
    {
        var outputs = new List<TensorOutput>();

        unsafe
        {
            var tensors = (CudaRsOvTensor*)tensorsPtr;
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

        CudaRsNative.OvFreeTensors(tensorsPtr, tensorCount);

        return new BackendResult { Outputs = outputs };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            CudaRsNative.OvDestroy(_handle);
            _disposed = true;
        }
    }
}

/// <summary>
/// OpenVINO inference options.
/// </summary>
public sealed class OpenVinoOptions
{
    /// <summary>Target device for inference.</summary>
    public CudaRsOvDevice Device { get; set; } = CudaRsOvDevice.Cpu;

    /// <summary>Device index when using GPU.X notation.</summary>
    public int DeviceIndex { get; set; } = 0;

    /// <summary>Number of inference streams (0 = auto).</summary>
    public int NumStreams { get; set; } = 0;

    /// <summary>Enable profiling for performance analysis.</summary>
    public bool EnableProfiling { get; set; } = false;
}

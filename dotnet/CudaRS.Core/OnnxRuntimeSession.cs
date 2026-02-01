using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CudaRS.Native;

namespace CudaRS.Core;

public sealed class OnnxTensor
{
    public float[] Data { get; init; } = Array.Empty<float>();
    public int[] Shape { get; init; } = Array.Empty<int>();
}

public sealed class OnnxRuntimeSession : SafeHandle
{
    public string ModelPath { get; }
    public int DeviceId { get; }

    public OnnxRuntimeSession(string modelPath, int deviceId = 0) : base(IntPtr.Zero, true)
    {
        ModelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        DeviceId = deviceId;

        CudaCheck.ThrowIfError(CudaRsNative.OnnxCreate(ModelPath, DeviceId, out var handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.OnnxDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    public unsafe IReadOnlyList<OnnxTensor> Run(float[] input, int[] shape)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (shape == null || shape.Length == 0)
            throw new ArgumentNullException(nameof(shape));

        var shape64 = new long[shape.Length];
        for (int i = 0; i < shape.Length; i++)
            shape64[i] = shape[i];

        fixed (float* inputPtr = input)
        fixed (long* shapePtr = shape64)
        {
            CudaCheck.ThrowIfError(CudaRsNative.OnnxRun(
                (ulong)handle.ToInt64(),
                inputPtr,
                (ulong)input.LongLength,
                shapePtr,
                (ulong)shape64.LongLength,
                out var tensorsPtr,
                out var tensorCount));

            try
            {
                return MarshalTensors(tensorsPtr, tensorCount);
            }
            finally
            {
                CudaRsNative.OnnxFreeTensors(tensorsPtr, tensorCount);
            }
        }
    }

    private static IReadOnlyList<OnnxTensor> MarshalTensors(IntPtr tensorsPtr, ulong tensorCount)
    {
        var results = new List<OnnxTensor>((int)tensorCount);
        var stride = Marshal.SizeOf<CudaRsTensor>();

        for (var i = 0; i < (int)tensorCount; i++)
        {
            var ptr = IntPtr.Add(tensorsPtr, i * stride);
            var tensor = Marshal.PtrToStructure<CudaRsTensor>(ptr);

            var data = new float[tensor.DataLen];
            if (tensor.Data != IntPtr.Zero && tensor.DataLen > 0)
                Marshal.Copy(tensor.Data, data, 0, (int)tensor.DataLen);

            var shape = new int[tensor.ShapeLen];
            if (tensor.Shape != IntPtr.Zero && tensor.ShapeLen > 0)
            {
                var shape64 = new long[tensor.ShapeLen];
                Marshal.Copy(tensor.Shape, shape64, 0, (int)tensor.ShapeLen);
                for (int s = 0; s < shape.Length; s++)
                    shape[s] = (int)shape64[s];
            }

            results.Add(new OnnxTensor { Data = data, Shape = shape });
        }

        return results;
    }
}

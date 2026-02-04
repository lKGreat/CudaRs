using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Interop;
using CudaRS.Native;

namespace CudaRS.OpenVino;

public sealed class OpenVinoPipeline : IDisposable
{
    private readonly PipelineHandle _handle;
    private readonly bool _ownsHandle;

    public OpenVinoPipeline(PipelineHandle handle, bool ownsHandle = true)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        _ownsHandle = ownsHandle;
    }

    public OpenVinoTensorOutput[] Run(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape)
    {
        if (input.IsEmpty)
            throw new ArgumentException("Input data required", nameof(input));
        if (shape.IsEmpty)
            throw new ArgumentException("Shape required", nameof(shape));

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
                SdkCheck.ThrowIfError(SdkNative.TensorPipelineRun(
                    _handle.Value,
                    dataPtr,
                    (nuint)inputSegment.Count,
                    shapePtrOffset,
                    (nuint)shapeSegment.Count));
            }
        }

        return ReadOutputs();
    }

    private OpenVinoTensorOutput[] ReadOutputs()
    {
        SdkCheck.ThrowIfError(SdkNative.PipelineGetOutputCount(_handle.Value, out var count));
        var outputs = new List<OpenVinoTensorOutput>((int)count);

        for (nuint i = 0; i < count; i++)
        {
            SdkCheck.ThrowIfError(SdkNative.PipelineGetOutputShapeLen(_handle.Value, i, out var shapeLen));
            var shape = new long[(int)shapeLen];
            unsafe
            {
                fixed (long* shapePtr = shape)
                {
                    SdkCheck.ThrowIfError(SdkNative.PipelineGetOutputShapeWrite(_handle.Value, i, shapePtr, shapeLen, out var written));
                    if (written != shapeLen)
                        throw new InvalidOperationException("Output shape length mismatch");
                }
            }

            SdkCheck.ThrowIfError(SdkNative.PipelineGetOutputBytes(_handle.Value, i, out var byteLen));
            var bytes = new byte[(int)byteLen];
            unsafe
            {
                fixed (byte* dst = bytes)
                {
                    SdkCheck.ThrowIfError(SdkNative.PipelineReadOutput(_handle.Value, i, dst, byteLen, out var written));
                    if (written != byteLen)
                        throw new InvalidOperationException("Output byte count mismatch");
                }
            }

            if (byteLen % sizeof(float) != 0)
                throw new InvalidOperationException("Output byte length not aligned to float");

            var floats = new float[byteLen / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);

            var intShape = new int[shape.Length];
            for (int s = 0; s < shape.Length; s++)
                intShape[s] = (int)shape[s];

            outputs.Add(new OpenVinoTensorOutput
            {
                Data = floats,
                Shape = intShape,
            });
        }

        return outputs.ToArray();
    }

    public void Dispose()
    {
        if (_ownsHandle)
            _handle.Dispose();
    }
}

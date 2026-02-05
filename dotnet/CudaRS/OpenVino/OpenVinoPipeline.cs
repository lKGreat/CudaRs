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

    public OpenVinoTensorOutput[][] RunBatch(ReadOnlyMemory<float>[] inputs, ReadOnlyMemory<long> singleShape)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("Batch inputs required", nameof(inputs));
        if (singleShape.IsEmpty)
            throw new ArgumentException("Shape required", nameof(singleShape));

        int batchSize = inputs.Length;

        // Verify all inputs have the same length
        long expectedSize = 1;
        foreach (var dim in singleShape.Span)
            expectedSize *= dim;

        foreach (var input in inputs)
        {
            if (input.Length != expectedSize)
                throw new ArgumentException($"All inputs must have size {expectedSize}", nameof(inputs));
        }

        if (!MemoryMarshal.TryGetArray(singleShape, out var shapeSegment))
            shapeSegment = new ArraySegment<long>(singleShape.ToArray());

        unsafe
        {
            // Prepare input arrays
            var inputPtrs = stackalloc float*[batchSize];
            var inputLens = stackalloc ulong[batchSize];
            var inputArrays = new float[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                if (!MemoryMarshal.TryGetArray(inputs[i], out var inputSegment))
                    inputArrays[i] = inputs[i].ToArray();
                else
                    inputArrays[i] = inputSegment.Array!;

                fixed (float* ptr = inputArrays[i])
                {
                    inputPtrs[i] = ptr;
                    inputLens[i] = (ulong)inputArrays[i].Length;
                }
            }

            CudaRsOvTensor** outBatchTensors;
            ulong* outBatchCounts;

            fixed (long* shapePtr = shapeSegment.Array)
            {
                var shapePtrOffset = shapePtr + shapeSegment.Offset;

                // Keep input arrays pinned during the call
                var handles = new GCHandle[batchSize];
                try
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        handles[i] = GCHandle.Alloc(inputArrays[i], GCHandleType.Pinned);
                        inputPtrs[i] = (float*)handles[i].AddrOfPinnedObject();
                    }

                    var result = SdkNative.OpenVinoRunBatch(
                        _handle.Value,
                        inputPtrs,
                        inputLens,
                        (ulong)batchSize,
                        shapePtrOffset,
                        (ulong)shapeSegment.Count,
                        out outBatchTensors,
                        out outBatchCounts);
                    ThrowIfFailed(result);
                }
                finally
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        if (handles[i].IsAllocated)
                            handles[i].Free();
                    }
                }
            }

            // Read outputs for each batch item
            var results = new OpenVinoTensorOutput[batchSize][];

            for (int b = 0; b < batchSize; b++)
            {
                var tensorCount = (int)outBatchCounts[b];
                var outputs = new List<OpenVinoTensorOutput>(tensorCount);

                for (int t = 0; t < tensorCount; t++)
                {
                    var tensor = outBatchTensors[b][t];
                    
                    // Copy data
                    var floats = new float[tensor.DataLen];
                    fixed (float* dst = floats)
                    {
                        Buffer.MemoryCopy((void*)tensor.Data, dst, floats.Length * sizeof(float), floats.Length * sizeof(float));
                    }

                    // Copy shape
                    var shape = new int[tensor.ShapeLen];
                    var shapePtr = (long*)tensor.Shape;
                    for (int s = 0; s < (int)tensor.ShapeLen; s++)
                    {
                        shape[s] = (int)shapePtr[s];
                    }

                    outputs.Add(new OpenVinoTensorOutput
                    {
                        Data = floats,
                        Shape = shape,
                    });
                }

                results[b] = outputs.ToArray();
            }

            // Free native memory
            var freeResult = SdkNative.OpenVinoFreeBatchTensors(outBatchTensors, outBatchCounts, (ulong)batchSize);
            ThrowIfFailed(freeResult);

            return results;
        }
    }

    private static void ThrowIfFailed(CudaRsResult result)
    {
        if (result != CudaRsResult.Success)
        {
            throw new InvalidOperationException($"OpenVINO batch operation failed with result: {result}");
        }
    }

    public void Dispose()
    {
        if (_ownsHandle)
            _handle.Dispose();
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Interop;
using CudaRS.Native;

namespace CudaRS.Yolo;

public sealed class YoloPipeline : IDisposable
{
    private readonly PipelineHandle _handle;
    private readonly bool _ownsHandle;

    public YoloPipeline(PipelineHandle handle, YoloConfig config, string modelId, bool ownsHandle = true)
    {
        _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        Config = config ?? throw new ArgumentNullException(nameof(config));
        ModelId = modelId ?? string.Empty;
        _ownsHandle = ownsHandle;
    }

    public string ModelId { get; }

    public YoloConfig Config { get; }

    public ModelInferenceResult Run(ReadOnlyMemory<byte> imageBytes, string channelId, long frameIndex = 0)
    {
        if (imageBytes.IsEmpty)
            throw new ArgumentException("Image bytes required", nameof(imageBytes));

        var meta = RunNative(imageBytes);
        var preprocess = new YoloPreprocessResult
        {
            Input = Array.Empty<float>(),
            InputShape = new[] { 1, Config.InputChannels, Config.InputHeight, Config.InputWidth },
            Scale = meta.Scale,
            PadX = meta.PadX,
            PadY = meta.PadY,
            OriginalWidth = meta.OriginalWidth,
            OriginalHeight = meta.OriginalHeight,
        };

        var backend = ReadOutputs();
        return YoloPostprocessor.Decode(ModelId, Config, backend, preprocess, channelId, frameIndex);
    }

    private SdkYoloPreprocessMeta RunNative(ReadOnlyMemory<byte> imageBytes)
    {
        if (!MemoryMarshal.TryGetArray(imageBytes, out var segment))
            segment = new ArraySegment<byte>(imageBytes.ToArray());

        unsafe
        {
            fixed (byte* ptr = segment.Array)
            {
                var dataPtr = ptr + segment.Offset;
                SdkCheck.ThrowIfError(SdkNative.YoloPipelineRunImage(_handle.Value, dataPtr, (nuint)segment.Count, out var meta));
                return meta;
            }
        }
    }

    private BackendResult ReadOutputs()
    {
        SdkCheck.ThrowIfError(SdkNative.PipelineGetOutputCount(_handle.Value, out var count));
        var outputs = new List<TensorOutput>((int)count);

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

            outputs.Add(new TensorOutput
            {
                Data = floats,
                Shape = intShape,
            });
        }

        return new BackendResult { Outputs = outputs };
    }

    public void Dispose()
    {
        if (_ownsHandle)
            _handle.Dispose();
    }
}

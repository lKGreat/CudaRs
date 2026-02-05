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

    /// <summary>
    /// Runs inference on encoded image bytes (JPEG/PNG).
    /// </summary>
    public ModelInferenceResult Run(ReadOnlyMemory<byte> encodedImageBytes, string channelId, long frameIndex = 0)
    {
        if (encodedImageBytes.IsEmpty)
            throw new ArgumentException("Encoded image bytes required", nameof(encodedImageBytes));

        var meta = RunNative(encodedImageBytes);
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

    public ModelInferenceResult Run(YoloEncodedImage image, string channelId, long frameIndex = 0)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return Run(image.Data, channelId, frameIndex);
    }

    private SdkYoloPreprocessMeta RunNative(ReadOnlyMemory<byte> encodedImageBytes)
    {
        if (!MemoryMarshal.TryGetArray(encodedImageBytes, out var segment))
            segment = new ArraySegment<byte>(encodedImageBytes.ToArray());

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

    public ModelInferenceResult[] RunBatch(ReadOnlyMemory<byte>[] encodedImages, string channelId, long startFrameIndex = 0)
    {
        if (encodedImages == null || encodedImages.Length == 0)
            throw new ArgumentException("Batch images required", nameof(encodedImages));

        int batchSize = encodedImages.Length;
        var results = new ModelInferenceResult[batchSize];

        unsafe
        {
            // Pin all image arrays
            var handles = new GCHandle[batchSize];
            var imagePtrs = stackalloc byte*[batchSize];
            var imageLens = stackalloc nuint[batchSize];
            var metas = stackalloc SdkYoloPreprocessMeta[batchSize];

            try
            {
                for (int i = 0; i < batchSize; i++)
                {
                    if (!MemoryMarshal.TryGetArray(encodedImages[i], out var segment))
                    {
                        var array = encodedImages[i].ToArray();
                        handles[i] = GCHandle.Alloc(array, GCHandleType.Pinned);
                        imagePtrs[i] = (byte*)handles[i].AddrOfPinnedObject();
                        imageLens[i] = (nuint)array.Length;
                    }
                    else
                    {
                        handles[i] = GCHandle.Alloc(segment.Array!, GCHandleType.Pinned);
                        imagePtrs[i] = (byte*)handles[i].AddrOfPinnedObject() + segment.Offset;
                        imageLens[i] = (nuint)segment.Count;
                    }
                }

                SdkCheck.ThrowIfError(SdkNative.YoloPipelineRunBatchImages(_handle.Value, imagePtrs, imageLens, (nuint)batchSize, metas));

                // For now, we only get outputs from the first image (as per Rust implementation)
                // Read shared outputs
                var backend = ReadOutputs();

                // Create results for each image using preprocessor metadata
                for (int i = 0; i < batchSize; i++)
                {
                    var preprocess = new YoloPreprocessResult
                    {
                        Input = Array.Empty<float>(),
                        InputShape = new[] { 1, Config.InputChannels, Config.InputHeight, Config.InputWidth },
                        Scale = metas[i].Scale,
                        PadX = metas[i].PadX,
                        PadY = metas[i].PadY,
                        OriginalWidth = metas[i].OriginalWidth,
                        OriginalHeight = metas[i].OriginalHeight,
                    };

                    results[i] = YoloPostprocessor.Decode(ModelId, Config, backend, preprocess, channelId, startFrameIndex + i);
                }
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

        return results;
    }

    public void Dispose()
    {
        if (_ownsHandle)
            _handle.Dispose();
    }
}

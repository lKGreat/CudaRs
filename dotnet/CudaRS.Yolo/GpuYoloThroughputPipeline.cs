using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Yolo;

/// <summary>
/// High-throughput YOLO pipeline:
/// decode (JPEG nvJPEG / PNG CPU in Rust) -> device HWC -> GPU letterbox -> TensorRT enqueue(device) -> D2H -> CPU postprocess.
/// 
/// Designed for stable throughput with multiple in-flight frames (multiple CUDA streams).
/// </summary>
public sealed class GpuYoloThroughputPipeline : IDisposable
{
    private readonly YoloModelDefinition _model;
    private readonly int _deviceId;
    private readonly int _workerCount;
    private readonly Channel<Job> _channel;
    private readonly CancellationTokenSource _cts = new();
    private readonly List<Task> _workers = new();

    private sealed record Job(ReadOnlyMemory<byte> Bytes, string ChannelId, long FrameIndex, TaskCompletionSource<ModelInferenceResult> Tcs);

    private sealed class WorkerState : IDisposable
    {
        public ulong Stream;
        public ulong Event;
        public ulong Decoder;
        public ulong Preprocess;
        public ulong Trt;

        public IntPtr OutputPinned;
        public ulong OutputBytes;
        public int[] OutputShape = Array.Empty<int>();
        public string ModelId = string.Empty;

        public void Dispose()
        {
            if (OutputPinned != IntPtr.Zero)
                _ = CudaRsNative.HostFreePinned(OutputPinned);

            if (Trt != 0)
                _ = CudaRsNative.TrtDestroy(Trt);

            if (Preprocess != 0)
                _ = CudaRsNative.PreprocessDestroy(Preprocess);

            if (Decoder != 0)
                _ = CudaRsNative.ImageDecoderDestroy(Decoder);

            if (Event != 0)
                _ = CudaRsNative.EventDestroy(Event);

            if (Stream != 0)
                _ = CudaRsNative.StreamDestroy(Stream);
        }
    }

    public GpuYoloThroughputPipeline(
        YoloModelDefinition model,
        int deviceId,
        int maxInputWidth,
        int maxInputHeight,
        int workerCount,
        int channelCapacity = 64)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _deviceId = deviceId;
        _workerCount = Math.Max(1, workerCount);

        if (_model.Config.Task != YoloTask.Detect)
            throw new NotSupportedException("GpuYoloThroughputPipeline currently supports Detect only.");

        _channel = Channel.CreateBounded<Job>(new BoundedChannelOptions(Math.Max(4, channelCapacity))
        {
            SingleReader = false,
            SingleWriter = false,
            FullMode = BoundedChannelFullMode.Wait,
        });

        // Create workers.
        for (var i = 0; i < _workerCount; i++)
        {
            var state = CreateWorkerState(maxInputWidth, maxInputHeight);
            _workers.Add(Task.Run(() => WorkerLoop(state, _cts.Token)));
        }
    }

    public ValueTask<ModelInferenceResult> EnqueueAsync(ReadOnlyMemory<byte> imageBytes, string channelId, long frameIndex, CancellationToken ct = default)
    {
        if (imageBytes.IsEmpty)
            throw new ArgumentException("Image bytes required", nameof(imageBytes));

        var tcs = new TaskCompletionSource<ModelInferenceResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        var job = new Job(imageBytes, channelId, frameIndex, tcs);

        if (!_channel.Writer.TryWrite(job))
        {
            return EnqueueSlowAsync(job, ct);
        }

        return new ValueTask<ModelInferenceResult>(tcs.Task);
    }

    private async ValueTask<ModelInferenceResult> EnqueueSlowAsync(Job job, CancellationToken ct)
    {
        await _channel.Writer.WriteAsync(job, ct).ConfigureAwait(false);
        return await job.Tcs.Task.ConfigureAwait(false);
    }

    private WorkerState CreateWorkerState(int maxInputWidth, int maxInputHeight)
    {
        // Ensure device selected.
        var r = CudaRsNative.DeviceSet(_deviceId);
        if (r != CudaRsResult.Success)
            throw new CudaException($"cuda device set failed: {r}");

        r = CudaRsNative.StreamCreate(out var stream);
        if (r != CudaRsResult.Success)
            throw new CudaException($"stream create failed: {r}");

        r = CudaRsNative.EventCreate(out var evt);
        if (r != CudaRsResult.Success)
        {
            _ = CudaRsNative.StreamDestroy(stream);
            throw new CudaException($"event create failed: {r}");
        }

        r = CudaRsNative.ImageDecoderCreate(out var decoder, maxInputWidth, maxInputHeight, _model.Config.InputChannels);
        if (r != CudaRsResult.Success)
            throw new CudaException($"image decoder create failed: {r}");

        r = CudaRsNative.PreprocessCreate(
            out var preprocess,
            _model.Config.InputWidth,
            _model.Config.InputHeight,
            _model.Config.InputChannels,
            maxInputWidth,
            maxInputHeight);
        if (r != CudaRsResult.Success)
            throw new CudaException($"preprocess create failed: {r} (did you build cudars_ffi with --features rtc?)");

        // One TRT engine handle per worker (avoid concurrent use of a single execution context).
        r = CudaRsNative.TrtLoadEngine(_model.ModelPath, _deviceId, out var trt);
        if (r != CudaRsResult.Success)
            throw new CudaException($"trt load engine failed: {r}");

        // Query output 0 shape + bytes.
        r = CudaRsNative.TrtGetOutputDevicePtr(trt, 0, out var outDevPtr, out var outBytes);
        if (r != CudaRsResult.Success)
            throw new CudaException($"trt get output device ptr failed: {r}");

        var shape = GetOutputShape(trt, 0);

        // Allocate pinned host output.
        r = CudaRsNative.HostAllocPinned(out var outHostPinned, (nuint)outBytes);
        if (r != CudaRsResult.Success)
            throw new CudaException($"host alloc pinned failed: {r}");

        return new WorkerState
        {
            Stream = stream,
            Event = evt,
            Decoder = decoder,
            Preprocess = preprocess,
            Trt = trt,
            OutputPinned = outHostPinned,
            OutputBytes = outBytes,
            OutputShape = shape,
            ModelId = _model.ModelId,
        };
    }

    private static int[] GetOutputShape(ulong trt, int index)
    {
        unsafe
        {
            Span<long> shape = stackalloc long[16];
            fixed (long* shapePtr = shape)
            {
                var res = CudaRsNative.TrtGetOutputInfo(trt, index, shapePtr, out var shapeLen, 16);
                if (res != CudaRsResult.Success || shapeLen <= 0)
                    return Array.Empty<int>();

                var arr = new int[shapeLen];
                for (int i = 0; i < shapeLen; i++)
                    arr[i] = (int)shape[i];
                return arr;
            }
        }
    }

    private async Task WorkerLoop(WorkerState state, CancellationToken ct)
    {
        try
        {
            while (await _channel.Reader.WaitToReadAsync(ct).ConfigureAwait(false))
            {
                while (_channel.Reader.TryRead(out var job))
                {
                    try
                    {
                        var result = RunOne(state, job.Bytes, job.ChannelId, job.FrameIndex);
                        job.Tcs.TrySetResult(result);
                    }
                    catch (Exception ex)
                    {
                        job.Tcs.TrySetResult(new ModelInferenceResult
                        {
                            ModelId = state.ModelId,
                            ChannelId = job.ChannelId,
                            FrameIndex = job.FrameIndex,
                            Success = false,
                            ErrorMessage = ex.Message,
                        });
                    }
                }
            }
        }
        finally
        {
            state.Dispose();
        }
    }

    private ModelInferenceResult RunOne(WorkerState state, ReadOnlyMemory<byte> bytes, string channelId, long frameIndex)
    {
        unsafe
        {
            fixed (byte* p = bytes.Span)
            {
                // Decode -> device HWC u8
                var r = CudaRsNative.ImageDecoderDecodeToDevice(
                    state.Decoder,
                    p,
                    (nuint)bytes.Length,
                    state.Stream,
                    out var devHwcPtr,
                    out var pitchBytes,
                    out var w,
                    out var h,
                    out var fmt);

                if (r != CudaRsResult.Success)
                    throw new CudaException($"decode failed: {r} (format={fmt})");

                // Preprocess (device) -> device CHW f32 (640x640)
                r = CudaRsNative.PreprocessRunDeviceOnStream(
                    state.Preprocess,
                    (byte*)devHwcPtr,
                    w,
                    h,
                    state.Stream,
                    0,
                    out var prep);

                if (r != CudaRsResult.Success)
                    throw new CudaException($"preprocess failed: {r}");

                var inputDevice = (float*)prep.OutputPtr;
                var inputLen = (ulong)(prep.OutputSize / sizeof(float));

                // TRT enqueue (device input)
                r = CudaRsNative.TrtEnqueueDevice(state.Trt, inputDevice, inputLen, state.Stream, 0);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"trt enqueue failed: {r}");

                // Copy output[0] to pinned host (async on the same stream)
                r = CudaRsNative.TrtGetOutputDevicePtr(state.Trt, 0, out var outDevPtr, out var outBytes);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"trt get output device ptr failed: {r}");

                r = CudaRsNative.MemcpyDtoHAsyncRaw((void*)state.OutputPinned, (void*)outDevPtr, (nuint)outBytes, state.Stream);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"dtoh async failed: {r}");

                // Record completion event and wait (worker thread blocks, GPU stream can overlap across workers)
                r = CudaRsNative.EventRecord(state.Event, state.Stream);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"event record failed: {r}");

                r = CudaRsNative.EventSynchronize(state.Event);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"event sync failed: {r}");

                var outFloats = (int)(state.OutputBytes / (ulong)sizeof(float));
                var span = new ReadOnlySpan<float>((void*)state.OutputPinned, outFloats);

                var preprocessMeta = new YoloPreprocessResult
                {
                    Input = Array.Empty<float>(),
                    InputShape = new[] { 1, _model.Config.InputChannels, _model.Config.InputHeight, _model.Config.InputWidth },
                    Scale = prep.Scale,
                    PadX = prep.PadX,
                    PadY = prep.PadY,
                    OriginalWidth = prep.OriginalWidth,
                    OriginalHeight = prep.OriginalHeight,
                };

                return YoloPostprocessor.DecodeDetectFromMainOutput(
                    state.ModelId,
                    _model.Config,
                    span,
                    state.OutputShape,
                    preprocessMeta,
                    channelId,
                    frameIndex);
            }
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _channel.Writer.TryComplete();

        try
        {
            Task.WaitAll(_workers.ToArray(), TimeSpan.FromSeconds(5));
        }
        catch
        {
            // ignore
        }

        _cts.Dispose();
    }
}

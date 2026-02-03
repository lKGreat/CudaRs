using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Yolo;

public sealed class GpuYoloThroughputOptions
{
    public int BatchSize { get; set; } = 1;
    public int MaxBatchDelayMs { get; set; } = 2;
    public bool AllowPartialBatch { get; set; } = true;
    public bool UseDedicatedThreads { get; set; } = false;
    public int[]? CpuAffinityCores { get; set; }
}

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
    private readonly GpuYoloThroughputOptions _options;
    private readonly int _batchSize;
    private readonly int _maxBatchDelayMs;
    private readonly bool _allowPartialBatch;
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

        public IntPtr InputDevice;
        public ulong InputBytes;
        public ulong InputBytesPerSample;
        public int[] InputShape = Array.Empty<int>();
        public int InputBatch;

        public IntPtr OutputDevice;
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
        : this(model, deviceId, maxInputWidth, maxInputHeight, workerCount, null, channelCapacity)
    {
    }

    public GpuYoloThroughputPipeline(
        YoloModelDefinition model,
        int deviceId,
        int maxInputWidth,
        int maxInputHeight,
        int workerCount,
        GpuYoloThroughputOptions? options,
        int channelCapacity = 64)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _deviceId = deviceId;
        _workerCount = Math.Max(1, workerCount);
        _options = options ?? new GpuYoloThroughputOptions();
        _batchSize = Math.Max(1, _options.BatchSize);
        _maxBatchDelayMs = Math.Max(0, _options.MaxBatchDelayMs);
        _allowPartialBatch = _options.AllowPartialBatch;

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
            _workers.Add(StartWorker(state));
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

        r = CudaRsNative.TrtGetInputDevicePtr(trt, 0, out var inDevPtr, out var inBytes);
        if (r != CudaRsResult.Success)
            throw new CudaException($"trt get input device ptr failed: {r}");

        var inputShape = GetInputShape(trt, 0);
        var inputBatch = inputShape.Length > 0 ? inputShape[0] : 1;
        if (inputBatch <= 0)
            throw new NotSupportedException("Dynamic batch size is not supported.");

        if (_batchSize != inputBatch)
            throw new NotSupportedException($"Batch size mismatch: engine batch={inputBatch}, configured batch={_batchSize}.");

        if (inBytes % (ulong)inputBatch != 0)
            throw new CudaException($"Input bytes {inBytes} not divisible by batch {inputBatch}.");

        var inputBytesPerSample = inBytes / (ulong)inputBatch;

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
            InputDevice = inDevPtr,
            InputBytes = inBytes,
            InputBytesPerSample = inputBytesPerSample,
            InputShape = inputShape,
            InputBatch = inputBatch,
            OutputDevice = outDevPtr,
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

    private static int[] GetInputShape(ulong trt, int index)
    {
        unsafe
        {
            Span<long> shape = stackalloc long[16];
            fixed (long* shapePtr = shape)
            {
                var res = CudaRsNative.TrtGetInputInfo(trt, index, shapePtr, out var shapeLen, 16);
                if (res != CudaRsResult.Success || shapeLen <= 0)
                    return Array.Empty<int>();

                var arr = new int[shapeLen];
                for (int i = 0; i < shapeLen; i++)
                    arr[i] = (int)shape[i];
                return arr;
            }
        }
    }

    private Task StartWorker(WorkerState state)
    {
        if (_options.UseDedicatedThreads)
        {
            return Task.Factory.StartNew(
                () => WorkerLoop(state, _cts.Token),
                _cts.Token,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
        }

        return Task.Run(() => WorkerLoop(state, _cts.Token), _cts.Token);
    }

    private void WorkerLoop(WorkerState state, CancellationToken ct)
    {
        try
        {
            if (_options.CpuAffinityCores is { Length: > 0 })
                ThreadAffinity.TrySetCurrentThreadAffinity(_options.CpuAffinityCores);

            while (true)
            {
                Job first;
                try
                {
                    first = _channel.Reader.ReadAsync(ct).AsTask().GetAwaiter().GetResult();
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (ChannelClosedException)
                {
                    break;
                }

                if (_batchSize <= 1)
                {
                    try
                    {
                        var result = RunOne(state, first.Bytes, first.ChannelId, first.FrameIndex);
                        first.Tcs.TrySetResult(result);
                    }
                    catch (Exception ex)
                    {
                        first.Tcs.TrySetResult(BuildErrorResult(state.ModelId, first.ChannelId, first.FrameIndex, ex));
                    }
                    continue;
                }

                var batch = CollectBatch(first, ct);
                try
                {
                    var results = RunBatch(state, batch);
                    for (int i = 0; i < batch.Count; i++)
                        batch[i].Tcs.TrySetResult(results[i]);
                }
                catch (Exception ex)
                {
                    foreach (var job in batch)
                        job.Tcs.TrySetResult(BuildErrorResult(state.ModelId, job.ChannelId, job.FrameIndex, ex));
                }
            }
        }
        finally
        {
            state.Dispose();
        }
    }

    private List<Job> CollectBatch(Job first, CancellationToken ct)
    {
        var jobs = new List<Job>(_batchSize) { first };
        if (_batchSize <= 1)
            return jobs;

        if (!_allowPartialBatch)
        {
            while (jobs.Count < _batchSize)
            {
                try
                {
                    jobs.Add(_channel.Reader.ReadAsync(ct).AsTask().GetAwaiter().GetResult());
                }
                catch (ChannelClosedException)
                {
                    break;
                }
            }

            return jobs;
        }

        var deadlineTicks = Stopwatch.GetTimestamp() + (long)(_maxBatchDelayMs * (Stopwatch.Frequency / 1000.0));
        while (jobs.Count < _batchSize)
        {
            if (_channel.Reader.TryRead(out var next))
            {
                jobs.Add(next);
                continue;
            }

            var remainingTicks = deadlineTicks - Stopwatch.GetTimestamp();
            if (remainingTicks <= 0)
                break;

            var remainingSeconds = remainingTicks / (double)Stopwatch.Frequency;
            if (remainingSeconds <= 0)
                break;

            var waitTask = _channel.Reader.WaitToReadAsync(ct).AsTask();
            var delayTask = Task.Delay(TimeSpan.FromSeconds(remainingSeconds), ct);
            var completed = Task.WhenAny(waitTask, delayTask).GetAwaiter().GetResult();

            if (completed == waitTask)
            {
                if (!waitTask.GetAwaiter().GetResult())
                    break;
                continue;
            }

            break;
        }

        return jobs;
    }

    private static ModelInferenceResult BuildErrorResult(string modelId, string channelId, long frameIndex, Exception ex)
    {
        return new ModelInferenceResult
        {
            ModelId = modelId,
            ChannelId = channelId,
            FrameIndex = frameIndex,
            Success = false,
            ErrorMessage = ex.Message,
        };
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

                // Preprocess (device) -> device CHW f32 (640x640) into TRT input buffer
                r = CudaRsNative.PreprocessRunDeviceOnStreamInto(
                    state.Preprocess,
                    (byte*)devHwcPtr,
                    w,
                    h,
                    state.Stream,
                    0,
                    (float*)state.InputDevice,
                    out var prep);

                if (r != CudaRsResult.Success)
                    throw new CudaException($"preprocess failed: {r}");

                var inputDevice = (float*)state.InputDevice;
                var inputLen = state.InputBytes / (ulong)sizeof(float);

                // TRT enqueue (device input)
                r = CudaRsNative.TrtEnqueueDevice(state.Trt, inputDevice, inputLen, state.Stream, 0);
                if (r != CudaRsResult.Success)
                    throw new CudaException($"trt enqueue failed: {r}");

                // Copy output[0] to pinned host (async on the same stream)
                r = CudaRsNative.MemcpyDtoHAsyncRaw((void*)state.OutputPinned, (void*)state.OutputDevice, (nuint)state.OutputBytes, state.Stream);
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

    private IReadOnlyList<ModelInferenceResult> RunBatch(WorkerState state, IReadOnlyList<Job> jobs)
    {
        if (jobs.Count == 0)
            return Array.Empty<ModelInferenceResult>();

        var batchSize = state.InputBatch;
        var actualCount = jobs.Count;
        var paddedJobs = jobs;

        if (actualCount < batchSize)
        {
            var list = new List<Job>(batchSize);
            list.AddRange(jobs);
            var last = jobs[actualCount - 1];
            while (list.Count < batchSize)
                list.Add(last);
            paddedJobs = list;
        }

        var preprocesses = new YoloPreprocessResult[batchSize];
        var channelIds = new string[batchSize];
        var frameIndices = new long[batchSize];

        unsafe
        {
            for (int i = 0; i < batchSize; i++)
            {
                var job = paddedJobs[i];
                channelIds[i] = job.ChannelId;
                frameIndices[i] = job.FrameIndex;

                fixed (byte* p = job.Bytes.Span)
                {
                    // Decode -> device HWC u8
                    var r = CudaRsNative.ImageDecoderDecodeToDevice(
                        state.Decoder,
                        p,
                        (nuint)job.Bytes.Length,
                        state.Stream,
                        out var devHwcPtr,
                        out var pitchBytes,
                        out var w,
                        out var h,
                        out var fmt);

                    if (r != CudaRsResult.Success)
                        throw new CudaException($"decode failed: {r} (format={fmt})");

                    var offsetBytes = (long)state.InputBytesPerSample * i;
                    var outputPtr = new IntPtr(state.InputDevice.ToInt64() + offsetBytes);

                    // Preprocess (device) -> device CHW f32 into batch input buffer
                    r = CudaRsNative.PreprocessRunDeviceOnStreamInto(
                        state.Preprocess,
                        (byte*)devHwcPtr,
                        w,
                        h,
                        state.Stream,
                        0,
                        (float*)outputPtr,
                        out var prep);

                    if (r != CudaRsResult.Success)
                        throw new CudaException($"preprocess failed: {r}");

                    if ((ulong)prep.OutputSize > state.InputBytesPerSample)
                        throw new CudaException($"preprocess output size {prep.OutputSize} exceeds input buffer {state.InputBytesPerSample}");

                    preprocesses[i] = new YoloPreprocessResult
                    {
                        Input = Array.Empty<float>(),
                        InputShape = new[] { 1, _model.Config.InputChannels, _model.Config.InputHeight, _model.Config.InputWidth },
                        Scale = prep.Scale,
                        PadX = prep.PadX,
                        PadY = prep.PadY,
                        OriginalWidth = prep.OriginalWidth,
                        OriginalHeight = prep.OriginalHeight,
                    };
                }
            }

            var inputDevice = (float*)state.InputDevice;
            var inputLen = state.InputBytes / (ulong)sizeof(float);

            // TRT enqueue (device input)
            var trtResult = CudaRsNative.TrtEnqueueDevice(state.Trt, inputDevice, inputLen, state.Stream, 0);
            if (trtResult != CudaRsResult.Success)
                throw new CudaException($"trt enqueue failed: {trtResult}");

            // Copy output[0] to pinned host (async on the same stream)
            var copyResult = CudaRsNative.MemcpyDtoHAsyncRaw((void*)state.OutputPinned, (void*)state.OutputDevice, (nuint)state.OutputBytes, state.Stream);
            if (copyResult != CudaRsResult.Success)
                throw new CudaException($"dtoh async failed: {copyResult}");

            // Record completion event and wait (worker thread blocks, GPU stream can overlap across workers)
            var evtResult = CudaRsNative.EventRecord(state.Event, state.Stream);
            if (evtResult != CudaRsResult.Success)
                throw new CudaException($"event record failed: {evtResult}");

            evtResult = CudaRsNative.EventSynchronize(state.Event);
            if (evtResult != CudaRsResult.Success)
                throw new CudaException($"event sync failed: {evtResult}");

            var outFloats = (int)(state.OutputBytes / (ulong)sizeof(float));
            var span = new ReadOnlySpan<float>((void*)state.OutputPinned, outFloats);

            var results = YoloPostprocessor.DecodeDetectBatch(
                state.ModelId,
                _model.Config,
                span,
                state.OutputShape,
                preprocesses,
                channelIds,
                frameIndices);

            if (actualCount == batchSize)
                return results;

            var trimmed = new ModelInferenceResult[actualCount];
            for (int i = 0; i < actualCount; i++)
                trimmed[i] = results[i];
            return trimmed;
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

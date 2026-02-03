using System;
using System.Threading;
using System.Threading.Tasks;

namespace CudaRS.Yolo;

public sealed class YoloGpuThroughputPipeline : IDisposable
{
    private readonly YoloPipeline _pipeline;
    private readonly SemaphoreSlim _semaphore;

    public YoloGpuThroughputPipeline(YoloModelBase model, YoloGpuThroughputOptions? options = null)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        var cfg = options ?? new YoloGpuThroughputOptions();
        _pipeline = model.CreatePipeline("default", cfg.PipelineOptions, ownsHandle: true);
        _semaphore = new SemaphoreSlim(Math.Max(1, cfg.MaxConcurrency));
    }

    public ValueTask<ModelInferenceResult> EnqueueAsync(
        ReadOnlyMemory<byte> encodedImageBytes,
        string channelId,
        long frameIndex,
        CancellationToken ct = default)
    {
        return new ValueTask<ModelInferenceResult>(RunAsync(encodedImageBytes, channelId, frameIndex, ct));
    }

    public ValueTask<ModelInferenceResult> EnqueueAsync(
        YoloEncodedImage image,
        string channelId,
        long frameIndex,
        CancellationToken ct = default)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return EnqueueAsync(image.Data, channelId, frameIndex, ct);
    }

    private async Task<ModelInferenceResult> RunAsync(
        ReadOnlyMemory<byte> encodedImageBytes,
        string channelId,
        long frameIndex,
        CancellationToken ct)
    {
        await _semaphore.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            return _pipeline.Run(encodedImageBytes, channelId, frameIndex);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public void Dispose()
    {
        _pipeline.Dispose();
        _semaphore.Dispose();
    }
}

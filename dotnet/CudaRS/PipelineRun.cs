using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace CudaRS;

public sealed class PipelineRun : IDisposable
{
    private readonly PipelineDefinition _definition;
    private readonly GpuMemoryMaintenance _memoryMaintenance;
    private readonly Dictionary<string, CudaRS.Core.MemoryPoolHandle> _memoryPools =
        new(StringComparer.OrdinalIgnoreCase);

    internal PipelineRun(PipelineDefinition definition)
    {
        _definition = definition;
        InitializeMemoryPools();
        _memoryMaintenance = new GpuMemoryMaintenance(definition.MemoryConfig);
        _memoryMaintenance.Start();
    }

    public PipelineDefinition Definition => _definition;

    public IReadOnlyDictionary<string, CudaRS.Core.MemoryPoolHandle> MemoryPools => _memoryPools;

    public DateTimeOffset LastDefragmentation => _memoryMaintenance.LastDefragmentation;

    public void StopMaintenance()
    {
        _memoryMaintenance.Dispose();
    }

    public void Dispose()
    {
        _memoryMaintenance.Dispose();
        foreach (var pool in _memoryPools.Values)
            pool.Dispose();
        _memoryPools.Clear();
    }

    private void InitializeMemoryPools()
    {
        foreach (var model in _definition.Models.Values)
        {
            var quota = new CudaRS.Native.CudaRsMemoryQuota
            {
                MaxBytes = (ulong)Math.Max(0, model.MemoryQuota.MaxBytes),
                PreallocateBytes = (ulong)Math.Max(0, model.MemoryQuota.PreallocateBytes),
                AllowFallbackToShared = model.MemoryQuota.AllowFallbackToShared,
                OomPolicy = model.MemoryQuota.OomPolicy switch
                {
                    OomPolicy.Wait => CudaRS.Native.CudaRsOomPolicy.Wait,
                    OomPolicy.Skip => CudaRS.Native.CudaRsOomPolicy.Skip,
                    OomPolicy.FallbackCpu => CudaRS.Native.CudaRsOomPolicy.FallbackCpu,
                    _ => CudaRS.Native.CudaRsOomPolicy.Fail,
                },
            };

            var deviceId = model.DeviceId ?? (_definition.MemoryConfig.DeviceIds.Length > 0
                ? _definition.MemoryConfig.DeviceIds[0]
                : 0);

            var pool = new CudaRS.Core.MemoryPoolHandle(model.ModelId, deviceId, quota);
            _memoryPools[model.ModelId] = pool;
        }

        if (_definition.MemoryConfig.DefragmentAsync == null)
        {
            _definition.MemoryConfig.DefragmentAsync = async token =>
            {
                foreach (var pool in _memoryPools.Values)
                {
                    token.ThrowIfCancellationRequested();
                    var stats = pool.GetStats();
                    if (stats.FragmentationRate >= _definition.MemoryConfig.Defragmentation.FragmentationThreshold)
                    {
                        pool.Defragment();
                    }
                }

                await Task.CompletedTask.ConfigureAwait(false);
            };
        }
    }

    public RunResult Run(IReadOnlyDictionary<string, object> inputs)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        return Run(PipelineInput.FromObjects(inputs));
    }

    public RunResult Run(PipelineInput input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        var stopwatch = Stopwatch.StartNew();
        var diagnostics = new List<string>();
        var outputs = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        var modelOutputs = new Dictionary<string, IReadOnlyDictionary<string, object>>(StringComparer.OrdinalIgnoreCase);

        foreach (var channel in _definition.Channels.Keys)
        {
            if (!input.Channels.TryGetValue(channel, out var payload))
            {
                diagnostics.Add($"Missing input for channel '{channel}'.");
                continue;
            }

            outputs[channel] = payload.Payload;
        }

        foreach (var model in _definition.Models.Values)
        {
            modelOutputs[model.ModelId] = new Dictionary<string, object>(outputs, StringComparer.OrdinalIgnoreCase);
        }

        stopwatch.Stop();

        return new RunResult
        {
            PipelineName = _definition.Name,
            Success = diagnostics.Count == 0,
            Elapsed = stopwatch.Elapsed,
            PerChannelOutputs = outputs,
            ModelOutputs = modelOutputs,
            Diagnostics = diagnostics,
        };
    }

    public Task<RunResult> RunAsync(IReadOnlyDictionary<string, object> inputs, CancellationToken cancellationToken = default)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        return RunAsync(PipelineInput.FromObjects(inputs), cancellationToken);
    }

    public Task<RunResult> RunAsync(PipelineInput input, CancellationToken cancellationToken = default)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (cancellationToken.IsCancellationRequested)
            return Task.FromCanceled<RunResult>(cancellationToken);

        return Task.Run(() => Run(input), cancellationToken);
    }

    public async Task RunStreamAsync(
        IAsyncEnumerable<PipelineInput> inputs,
        Func<RunResult, Task>? onResult = null,
        CancellationToken cancellationToken = default)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        onResult ??= _ => Task.CompletedTask;

        using var limiter = new SemaphoreSlim(_definition.Execution.MaxConcurrency, _definition.Execution.MaxConcurrency);
        var running = new List<Task>();

        await foreach (var input in inputs.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            await limiter.WaitAsync(cancellationToken).ConfigureAwait(false);

            var task = Task.Run(async () =>
            {
                try
                {
                    var result = Run(input);
                    await onResult(result).ConfigureAwait(false);
                }
                finally
                {
                    limiter.Release();
                }
            }, cancellationToken);

            running.Add(task);

            running.RemoveAll(t => t.IsCompleted);
        }

        await Task.WhenAll(running).ConfigureAwait(false);
    }
}

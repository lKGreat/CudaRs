using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Threading;
using System.Threading.Tasks;

namespace CudaRS;

public sealed class PipelineRun
{
    private readonly PipelineDefinition _definition;

    internal PipelineRun(PipelineDefinition definition)
    {
        _definition = definition;
    }

    public PipelineDefinition Definition => _definition;

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

        foreach (var channel in _definition.Channels.Keys)
        {
            if (!input.Channels.TryGetValue(channel, out var payload))
            {
                diagnostics.Add($"Missing input for channel '{channel}'.");
                continue;
            }

            outputs[channel] = payload.Payload;
        }

        stopwatch.Stop();

        return new RunResult
        {
            PipelineName = _definition.Name,
            Success = diagnostics.Count == 0,
            Elapsed = stopwatch.Elapsed,
            PerChannelOutputs = outputs,
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

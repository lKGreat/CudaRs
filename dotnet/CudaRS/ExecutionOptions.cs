using System;

namespace CudaRS;

public sealed class ExecutionOptions
{
    public int MaxConcurrency { get; private set; } = 1;
    public int TimeoutMs { get; private set; } = 30_000;
    public int RetryCount { get; private set; } = 0;
    public int MaxQueueDepth { get; private set; } = 5_000;
    public BackpressurePolicy Backpressure { get; private set; } = BackpressurePolicy.DropLowestPriority;
    public StreamMode StreamMode { get; private set; } = StreamMode.Async;
    public int StreamPoolSize { get; private set; } = Math.Min(64, Environment.ProcessorCount * 8);

    public ExecutionOptions WithMaxConcurrency(int value)
    {
        MaxConcurrency = Math.Max(1, value);
        return this;
    }

    public ExecutionOptions WithTimeoutMs(int value)
    {
        TimeoutMs = Math.Max(1, value);
        return this;
    }

    public ExecutionOptions WithRetry(int value)
    {
        RetryCount = Math.Max(0, value);
        return this;
    }

    public ExecutionOptions WithMaxQueueDepth(int value)
    {
        MaxQueueDepth = Math.Max(1, value);
        return this;
    }

    public ExecutionOptions WithBackpressurePolicy(BackpressurePolicy policy)
    {
        Backpressure = policy;
        return this;
    }

    public ExecutionOptions WithStreamMode(StreamMode mode)
    {
        StreamMode = mode;
        return this;
    }

    public ExecutionOptions WithStreamPoolSize(int value)
    {
        StreamPoolSize = Math.Max(1, value);
        return this;
    }
}

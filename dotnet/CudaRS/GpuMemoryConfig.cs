using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace CudaRS;

public enum OomPolicy
{
    Fail = 0,
    Wait = 1,
    Skip = 2,
    FallbackCpu = 3,
}

public enum GpuDeviceSelection
{
    Auto = 0,
    Fixed = 1,
    RoundRobin = 2,
}

public sealed class MemoryQuota
{
    public long MaxBytes { get; internal set; } = 0;
    public long PreallocateBytes { get; internal set; } = 0;
    public bool AllowFallbackToShared { get; internal set; } = false;
    public OomPolicy OomPolicy { get; internal set; } = OomPolicy.Fail;

    public static MemoryQuota Unlimited => new();

    public static MemoryQuota FromMb(int mb)
        => new MemoryQuota { MaxBytes = Math.Max(0, mb) * 1024L * 1024L };
}

public sealed class MemoryDefragmentationOptions
{
    public bool Enabled { get; internal set; } = true;
    public TimeSpan Interval { get; internal set; } = TimeSpan.FromMinutes(10);
    public float FragmentationThreshold { get; internal set; } = 0.3f;
}

public sealed class GpuMemoryConfig
{
    public int[] DeviceIds { get; internal set; } = new[] { 0 };
    public GpuDeviceSelection DeviceSelection { get; internal set; } = GpuDeviceSelection.RoundRobin;
    public long ReservedBytes { get; internal set; } = 256 * 1024 * 1024L;
    public long SharedPoolBytes { get; internal set; } = 512 * 1024 * 1024L;
    public bool EnableDefragmentation { get; internal set; } = true;
    public MemoryDefragmentationOptions Defragmentation { get; internal set; } = new MemoryDefragmentationOptions();

    /// <summary>
    /// Optional callback invoked for defragmentation.
    /// </summary>
    public Func<CancellationToken, Task>? DefragmentAsync { get; internal set; }
}

public sealed class GpuMemoryStats
{
    public int DeviceId { get; init; }
    public long TotalMemory { get; init; }
    public long FreeMemory { get; init; }
    public long UsedMemory => TotalMemory - FreeMemory;
    public float UtilizationRate => TotalMemory > 0 ? (float)UsedMemory / TotalMemory : 0f;
    public DateTimeOffset LastDefragmentation { get; internal set; }
}

internal sealed class GpuMemoryMaintenance : IDisposable
{
    private readonly GpuMemoryConfig _config;
    private readonly CancellationTokenSource _cts = new();
    private Timer? _timer;
    private DateTimeOffset _lastDefrag = DateTimeOffset.MinValue;

    public GpuMemoryMaintenance(GpuMemoryConfig config)
    {
        _config = config;
    }

    public DateTimeOffset LastDefragmentation => _lastDefrag;

    public void Start()
    {
        if (!_config.EnableDefragmentation || !_config.Defragmentation.Enabled)
            return;

        var interval = _config.Defragmentation.Interval;
        if (interval <= TimeSpan.Zero)
            interval = TimeSpan.FromMinutes(10);

        _timer ??= new Timer(OnTimer, null, interval, interval);
    }

    private async void OnTimer(object? state)
    {
        try
        {
            if (_cts.IsCancellationRequested)
                return;

            if (_config.DefragmentAsync != null)
                await _config.DefragmentAsync(_cts.Token).ConfigureAwait(false);

            _lastDefrag = DateTimeOffset.UtcNow;
        }
        catch
        {
            // Best-effort maintenance; errors are ignored by design.
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _timer?.Dispose();
        _cts.Dispose();
    }
}

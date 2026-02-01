using System;

namespace CudaRS;

public sealed class ModelBuilder
{
    private readonly ModelOptions _options;

    internal ModelBuilder(ModelOptions options)
    {
        _options = options;
    }

    public ModelBuilder FromPath(string path)
    {
        _options.ModelPath = path ?? string.Empty;
        return this;
    }

    public ModelBuilder WithBackend(string backend)
    {
        _options.Backend = backend ?? "auto";
        return this;
    }

    public ModelBuilder OnDevice(string device)
    {
        _options.Device = device ?? "auto";
        return this;
    }

    public ModelBuilder WithDeviceId(int deviceId)
    {
        _options.DeviceId = deviceId;
        return this;
    }

    public ModelBuilder WithPrecision(string precision)
    {
        _options.Precision = precision ?? "auto";
        return this;
    }

    public ModelBuilder WithWorkspaceMb(int mb)
    {
        _options.WorkspaceMb = Math.Max(16, mb);
        return this;
    }

    public ModelBuilder WithMemoryQuota(Action<MemoryQuota> configure)
    {
        configure?.Invoke(_options.MemoryQuota);
        return this;
    }
}

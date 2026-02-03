using System;
using System.Collections.Generic;

namespace CudaRS;

public sealed class RunResult
{
    public string PipelineName { get; internal set; } = string.Empty;
    public bool Success { get; internal set; }
    public TimeSpan Elapsed { get; internal set; }
    public IReadOnlyDictionary<string, object> PerChannelOutputs { get; internal set; } =
        new Dictionary<string, object>();
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, object>> ModelOutputs { get; internal set; } =
        new Dictionary<string, IReadOnlyDictionary<string, object>>();
    public IReadOnlyList<string> Diagnostics { get; internal set; } = Array.Empty<string>();
}

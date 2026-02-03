using System;
using System.Collections.Generic;

namespace CudaRS;

public sealed class PipelineInput
{
    public PipelineInput(IReadOnlyDictionary<string, ChannelInput> channels)
    {
        Channels = channels ?? throw new ArgumentNullException(nameof(channels));
    }

    public IReadOnlyDictionary<string, ChannelInput> Channels { get; }

    public static PipelineInput FromObjects(IReadOnlyDictionary<string, object> channels)
    {
        if (channels == null)
            throw new ArgumentNullException(nameof(channels));

        var mapped = new Dictionary<string, ChannelInput>(StringComparer.OrdinalIgnoreCase);
        foreach (var (name, payload) in channels)
            mapped[name] = new ChannelInput(payload);

        return new PipelineInput(mapped);
    }
}

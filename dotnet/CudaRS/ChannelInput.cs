using System;

namespace CudaRS;

public sealed class ChannelInput
{
    public ChannelInput(object payload)
    {
        Payload = payload ?? throw new ArgumentNullException(nameof(payload));
    }

    public object Payload { get; }
}

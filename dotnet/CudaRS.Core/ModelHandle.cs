namespace CudaRS.Interop;

public readonly struct ModelHandle
{
    public ModelHandle(ulong value)
    {
        Value = value;
    }

    public ulong Value { get; }
}

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CudaRS.Interop;

public sealed class Utf8Buffer : IDisposable
{
    private GCHandle _handle;

    public Utf8Buffer(string? value)
    {
        var bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
        Bytes = bytes;
        _handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
    }

    public byte[] Bytes { get; }

    public IntPtr Pointer => _handle.AddrOfPinnedObject();

    public nuint Length => (nuint)Bytes.Length;

    public void Dispose()
    {
        if (_handle.IsAllocated)
            _handle.Free();
    }
}

using System;
using System.Runtime.InteropServices;
using CudaRS.Native;

namespace CudaRS.Core;

/// <summary>
/// SafeHandle wrapper for CUDA streams.
/// </summary>
public sealed class CudaStream : SafeHandle
{
    public CudaStream() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.StreamCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    internal CudaStream(ulong handle) : base(new IntPtr((long)handle), true)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.StreamDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Synchronize the stream.
    /// </summary>
    public void Synchronize()
    {
        CudaCheck.ThrowIfError(CudaRsNative.StreamSynchronize((ulong)handle.ToInt64()));
    }

    /// <summary>
    /// Get the native handle value.
    /// </summary>
    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for CUDA events.
/// </summary>
public sealed class CudaEvent : SafeHandle
{
    public CudaEvent() : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.EventCreate(out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.EventDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Record the event on a stream.
    /// </summary>
    public void Record(CudaStream stream)
    {
        CudaCheck.ThrowIfError(CudaRsNative.EventRecord((ulong)handle.ToInt64(), stream.Handle));
    }

    /// <summary>
    /// Synchronize the event.
    /// </summary>
    public void Synchronize()
    {
        CudaCheck.ThrowIfError(CudaRsNative.EventSynchronize((ulong)handle.ToInt64()));
    }

    /// <summary>
    /// Get elapsed time between this event and a start event.
    /// </summary>
    public float ElapsedTime(CudaEvent start)
    {
        CudaCheck.ThrowIfError(CudaRsNative.EventElapsedTime(out float ms, start.Handle, (ulong)handle.ToInt64()));
        return ms;
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for CUDA device buffers.
/// </summary>
public sealed class CudaBuffer : SafeHandle
{
    public nuint Size { get; }

    public CudaBuffer(nuint size) : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.Malloc(out ulong handle, size));
        SetHandle(new IntPtr((long)handle));
        Size = size;
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.Free((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Copy data from host to device.
    /// </summary>
    public unsafe void CopyFromHost(ReadOnlySpan<byte> data)
    {
        fixed (byte* ptr = data)
        {
            CudaCheck.ThrowIfError(CudaRsNative.MemcpyHtoD((ulong)handle.ToInt64(), ptr, (nuint)data.Length));
        }
    }

    /// <summary>
    /// Copy data from device to host.
    /// </summary>
    public unsafe void CopyToHost(Span<byte> data)
    {
        fixed (byte* ptr = data)
        {
            CudaCheck.ThrowIfError(CudaRsNative.MemcpyDtoH(ptr, (ulong)handle.ToInt64(), (nuint)data.Length));
        }
    }

    /// <summary>
    /// Set all bytes to a value.
    /// </summary>
    public void Memset(int value)
    {
        CudaCheck.ThrowIfError(CudaRsNative.Memset((ulong)handle.ToInt64(), value));
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

/// <summary>
/// SafeHandle wrapper for CUDA driver contexts.
/// </summary>
public sealed class CudaContext : SafeHandle
{
    public CudaContext(int device) : base(IntPtr.Zero, true)
    {
        CudaCheck.ThrowIfError(CudaRsNative.ContextCreate(out ulong handle, device));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.ContextDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    /// <summary>
    /// Synchronize the context.
    /// </summary>
    public void Synchronize()
    {
        CudaCheck.ThrowIfError(CudaRsNative.ContextSynchronize((ulong)handle.ToInt64()));
    }

    public ulong Handle => (ulong)handle.ToInt64();
}

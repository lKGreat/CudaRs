using CudaRS.Core;
using System.Runtime.CompilerServices;

namespace CudaRS;

/// <summary>
/// Generic device buffer for typed data.
/// </summary>
/// <typeparam name="T">The element type (must be unmanaged).</typeparam>
public sealed class DeviceBuffer<T> : IDisposable where T : unmanaged
{
    private readonly CudaBuffer _buffer;
    private bool _disposed;

    /// <summary>
    /// The number of elements.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Create a device buffer with the specified number of elements.
    /// </summary>
    public DeviceBuffer(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        Length = length;
        _buffer = new CudaBuffer((nuint)(length * Unsafe.SizeOf<T>()));
    }

    /// <summary>
    /// Create a device buffer from host data.
    /// </summary>
    public DeviceBuffer(ReadOnlySpan<T> data) : this(data.Length)
    {
        CopyFromHost(data);
    }

    /// <summary>
    /// Copy data from host to device.
    /// </summary>
    public void CopyFromHost(ReadOnlySpan<T> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length > Length)
            throw new ArgumentException("Data exceeds buffer length", nameof(data));
        
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data);
        _buffer.CopyFromHost(bytes);
    }

    /// <summary>
    /// Copy data from device to host.
    /// </summary>
    public void CopyToHost(Span<T> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length > Length)
            throw new ArgumentException("Data exceeds buffer length", nameof(data));
        
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data);
        _buffer.CopyToHost(bytes);
    }

    /// <summary>
    /// Copy all data to a new array.
    /// </summary>
    public T[] ToArray()
    {
        var result = new T[Length];
        CopyToHost(result);
        return result;
    }

    /// <summary>
    /// Set all bytes to zero.
    /// </summary>
    public void Clear() => _buffer.Memset(0);

    /// <summary>
    /// Get the underlying buffer handle.
    /// </summary>
    internal CudaBuffer Buffer => _buffer;

    public void Dispose()
    {
        if (!_disposed)
        {
            _buffer.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>
/// Extension methods for device buffers.
/// </summary>
public static class DeviceBufferExtensions
{
    /// <summary>
    /// Create a device buffer from an array.
    /// </summary>
    public static DeviceBuffer<T> ToDevice<T>(this T[] data) where T : unmanaged
        => new DeviceBuffer<T>(data);

    /// <summary>
    /// Create a device buffer from a span.
    /// </summary>
    public static DeviceBuffer<T> ToDevice<T>(this ReadOnlySpan<T> data) where T : unmanaged
        => new DeviceBuffer<T>(data);
}

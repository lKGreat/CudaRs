using System;
using System.Runtime.InteropServices;
using CudaRS.Native;

namespace CudaRS.Core;

/// <summary>
/// SafeHandle wrapper for a GPU memory pool.
/// </summary>
public sealed class MemoryPoolHandle : SafeHandle
{
    public string PoolId { get; }
    public int DeviceId { get; }

    public MemoryPoolHandle(string poolId, CudaRsMemoryQuota quota) : base(IntPtr.Zero, true)
    {
        PoolId = poolId ?? throw new ArgumentNullException(nameof(poolId));
        DeviceId = -1;
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolCreate(PoolId, quota, out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public MemoryPoolHandle(string poolId, int deviceId, CudaRsMemoryQuota quota) : base(IntPtr.Zero, true)
    {
        PoolId = poolId ?? throw new ArgumentNullException(nameof(poolId));
        DeviceId = deviceId;
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolCreateWithDevice(PoolId, deviceId, quota, out ulong handle));
        SetHandle(new IntPtr((long)handle));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return CudaRsNative.MemoryPoolDestroy((ulong)handle.ToInt64()) == CudaRsResult.Success;
    }

    public ulong Allocate(ulong size)
    {
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolAllocate((ulong)handle.ToInt64(), size, out var ptr));
        return ptr;
    }

    public void Free(ulong ptr)
    {
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolFree((ulong)handle.ToInt64(), ptr));
    }

    public CudaRsMemoryPoolStats GetStats()
    {
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolGetStats((ulong)handle.ToInt64(), out var stats));
        return stats;
    }

    public void Defragment()
    {
        CudaCheck.ThrowIfError(CudaRsNative.MemoryPoolDefragment((ulong)handle.ToInt64()));
    }
}

/// <summary>
/// GPU memory statistics helper.
/// </summary>
public static class GpuMemory
{
    public static CudaRsGpuMemoryStats GetStats(int deviceId)
    {
        CudaCheck.ThrowIfError(CudaRsNative.GpuGetMemoryStats(deviceId, out var stats));
        return stats;
    }
}

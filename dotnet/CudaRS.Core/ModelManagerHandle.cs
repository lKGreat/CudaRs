using System;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Interop;

public sealed class ModelManagerHandle : SafeHandle
{
    public ModelManagerHandle() : base(IntPtr.Zero, true)
    {
        SdkCheck.ThrowIfError(SdkNative.ModelManagerCreate(out var handle));
        SetHandle(new IntPtr(unchecked((long)handle)));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return SdkNative.ModelManagerDestroy(unchecked((ulong)handle.ToInt64())) == SdkErr.Ok;
    }

    public ulong Value => unchecked((ulong)handle.ToInt64());
}

using System;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS.Interop;

public sealed class PipelineHandle : SafeHandle
{
    public PipelineHandle(ulong handleValue) : base(IntPtr.Zero, true)
    {
        SetHandle(new IntPtr(unchecked((long)handleValue)));
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        return SdkNative.PipelineDestroy(unchecked((ulong)handle.ToInt64())) == SdkErr.Ok;
    }

    public ulong Value => unchecked((ulong)handle.ToInt64());
}

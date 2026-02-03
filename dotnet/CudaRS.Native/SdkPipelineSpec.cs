using System;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

[StructLayout(LayoutKind.Sequential)]
public struct SdkPipelineSpec
{
    public IntPtr IdPtr;
    public nuint IdLen;
    public SdkPipelineKind Kind;
    public IntPtr ConfigJsonPtr;
    public nuint ConfigJsonLen;
}

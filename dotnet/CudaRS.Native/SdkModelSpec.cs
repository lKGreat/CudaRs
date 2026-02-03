using System;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

[StructLayout(LayoutKind.Sequential)]
public struct SdkModelSpec
{
    public IntPtr IdPtr;
    public nuint IdLen;
    public SdkModelKind Kind;
    public IntPtr ConfigJsonPtr;
    public nuint ConfigJsonLen;
}

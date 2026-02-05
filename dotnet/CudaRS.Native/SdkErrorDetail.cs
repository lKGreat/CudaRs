using System;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

[StructLayout(LayoutKind.Sequential)]
public struct SdkErrorDetail
{
    public SdkErr Code;
    public IntPtr MessagePtr;
    public nuint MessageLen;
    public IntPtr MissingFilePtr;
    public nuint MissingFileLen;
    public IntPtr SearchPathsPtr;
    public nuint SearchPathsLen;
    public IntPtr SuggestionPtr;
    public nuint SuggestionLen;
}

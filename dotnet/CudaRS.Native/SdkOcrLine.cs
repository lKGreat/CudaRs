using System.Runtime.InteropServices;

namespace CudaRS.Native;

[StructLayout(LayoutKind.Sequential)]
public unsafe struct SdkOcrLine
{
    public fixed float Points[8];
    public float Score;
    public int ClassLabel;
    public float ClassScore;
    public uint TextOffset;
    public uint TextLen;
}

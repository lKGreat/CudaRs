using System.Runtime.InteropServices;

namespace CudaRS.Native;

[StructLayout(LayoutKind.Sequential)]
public struct SdkYoloPreprocessMeta
{
    public float Scale;
    public int PadX;
    public int PadY;
    public int OriginalWidth;
    public int OriginalHeight;
}

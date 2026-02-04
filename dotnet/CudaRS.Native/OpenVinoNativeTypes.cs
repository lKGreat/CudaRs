using System;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

public enum CudaRsOvDevice
{
    Cpu = 0,
    Gpu = 1,
    GpuIndex = 2,
    Npu = 3,
    Auto = 4,
}

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvConfig
{
    public CudaRsOvDevice Device;
    public int DeviceIndex;
    public int NumStreams;
    public int EnableProfiling;
    public IntPtr PropertiesJsonPtr;
    public nuint PropertiesJsonLen;
}

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvTensor
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

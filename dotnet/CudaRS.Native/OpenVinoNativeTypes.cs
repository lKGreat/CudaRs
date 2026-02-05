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
public struct CudaRsOvConfigV2
{
    public uint StructSize;
    public CudaRsOvDevice Device;
    public int DeviceIndex;
    public IntPtr DeviceNamePtr;
    public nuint DeviceNameLen;
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

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvTensorInfo
{
    public IntPtr NamePtr;
    public ulong NameLen;
    public IntPtr Shape;
    public ulong ShapeLen;
    public int ElementType;
}

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvPartialDim
{
    public int IsStatic;
    public long Value;
}

[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvPartialShapeArray
{
    public IntPtr Dims;
    public ulong Rank;
}

using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS;

/// <summary>
/// High-level CUDA device management.
/// </summary>
public static class Cuda
{
    /// <summary>
    /// Get the number of CUDA devices.
    /// </summary>
    public static int DeviceCount
    {
        get
        {
            CudaCheck.ThrowIfError(CudaRsNative.DeviceGetCount(out int count));
            return count;
        }
    }

    /// <summary>
    /// Get or set the current device.
    /// </summary>
    public static int CurrentDevice
    {
        get
        {
            CudaCheck.ThrowIfError(CudaRsNative.DeviceGet(out int device));
            return device;
        }
        set
        {
            CudaCheck.ThrowIfError(CudaRsNative.DeviceSet(value));
        }
    }

    /// <summary>
    /// Synchronize the current device.
    /// </summary>
    public static void Synchronize()
    {
        CudaCheck.ThrowIfError(CudaRsNative.DeviceSynchronize());
    }

    /// <summary>
    /// Reset the current device.
    /// </summary>
    public static void Reset()
    {
        CudaCheck.ThrowIfError(CudaRsNative.DeviceReset());
    }

    /// <summary>
    /// Get the CudaRS library version.
    /// </summary>
    public static string Version
    {
        get
        {
            var ptr = CudaRsNative.GetVersion();
            return System.Runtime.InteropServices.Marshal.PtrToStringAnsi(ptr) ?? "unknown";
        }
    }
}

/// <summary>
/// High-level CUDA driver operations.
/// </summary>
public static class CudaDriver
{
    private static bool _initialized = false;

    /// <summary>
    /// Initialize the CUDA driver.
    /// </summary>
    public static void Initialize()
    {
        if (!_initialized)
        {
            CudaCheck.ThrowIfError(CudaRsNative.DriverInit());
            _initialized = true;
        }
    }

    /// <summary>
    /// Get the CUDA driver version.
    /// </summary>
    public static int Version
    {
        get
        {
            Initialize();
            CudaCheck.ThrowIfError(CudaRsNative.DriverGetVersion(out int version));
            return version;
        }
    }
}

/// <summary>
/// High-level NVML operations for GPU management.
/// </summary>
public static class GpuManagement
{
    private static bool _initialized = false;

    /// <summary>
    /// Initialize NVML.
    /// </summary>
    public static void Initialize()
    {
        if (!_initialized)
        {
            CudaCheck.ThrowIfError(CudaRsNative.NvmlInit());
            _initialized = true;
        }
    }

    /// <summary>
    /// Shutdown NVML.
    /// </summary>
    public static void Shutdown()
    {
        if (_initialized)
        {
            CudaCheck.ThrowIfError(CudaRsNative.NvmlShutdown());
            _initialized = false;
        }
    }

    /// <summary>
    /// Get the number of GPUs.
    /// </summary>
    public static uint DeviceCount
    {
        get
        {
            Initialize();
            CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetCount(out uint count));
            return count;
        }
    }

    /// <summary>
    /// Get memory info for a GPU.
    /// </summary>
    public static CudaRsMemoryInfo GetMemoryInfo(uint index)
    {
        Initialize();
        CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetMemoryInfo(index, out var info));
        return info;
    }

    /// <summary>
    /// Get utilization rates for a GPU.
    /// </summary>
    public static CudaRsUtilizationRates GetUtilizationRates(uint index)
    {
        Initialize();
        CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetUtilizationRates(index, out var rates));
        return rates;
    }

    /// <summary>
    /// Get temperature for a GPU (in Celsius).
    /// </summary>
    public static uint GetTemperature(uint index)
    {
        Initialize();
        CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetTemperature(index, out uint temp));
        return temp;
    }

    /// <summary>
    /// Get power usage for a GPU (in milliwatts).
    /// </summary>
    public static uint GetPowerUsage(uint index)
    {
        Initialize();
        CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetPowerUsage(index, out uint power));
        return power;
    }

    /// <summary>
    /// Get fan speed for a GPU (percentage).
    /// </summary>
    public static uint GetFanSpeed(uint index)
    {
        Initialize();
        CudaCheck.ThrowIfError(CudaRsNative.NvmlDeviceGetFanSpeed(index, out uint speed));
        return speed;
    }
}

/// <summary>
/// NVRTC compiler operations.
/// </summary>
public static class Nvrtc
{
    /// <summary>
    /// Get the NVRTC version.
    /// </summary>
    public static (int Major, int Minor) Version
    {
        get
        {
            CudaCheck.ThrowIfError(CudaRsNative.NvrtcVersion(out int major, out int minor));
            return (major, minor);
        }
    }
}
